# coding=gbk
from IsolationForest import PyTorchIsolationForest
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig
from sklearn.decomposition import PCA

class JailbreakDetector:
    def __init__(self, model, tokenizer, background_layered_activations, all_labels, your_flag,
                 n_estimators=100, random_state=42, k_nb=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.your_flag = your_flag
        self.k_nb = k_nb
        
        # === HYPERBOLIC MTK CHANGE 1: PCA Whitening & Lorentz Mapping ===
        # Here we fix the "Narrow Cone" anisotropy issue by applying PCA Whitening
        # and then projecting the vectors into the Lorentz Hyperboloid model.
        print("Applying PCA Whitening and mapping to Lorentz space...")
        self.num_layers = len(background_layered_activations[0]) if isinstance(background_layered_activations, list) else background_layered_activations.shape[1]
        
        self.pca_models = []
        pca_dim = 64 # Reduce from 4096 to 64
        
        # We need to map shape (1600, 32, 4096) to (1600, 32, pca_dim + 1)
        num_samples = background_layered_activations.shape[0]
        lorentz_activations = torch.empty((num_samples, self.num_layers, pca_dim + 1), device=self.device)
        
        for l in range(self.num_layers):
            pca = PCA(n_components=pca_dim, whiten=False)
            # Fit PCA on this layer's data
            layer_data = background_layered_activations[:, l, :].cpu().numpy()
            layer_pca = pca.fit_transform(layer_data)
            self.pca_models.append(pca)
            
            # Apply scale factor and map to Lorentz space (add time coordinate t = sqrt(1 + ||x||^2))
            layer_scaled = layer_pca * 15.0
            layer_scaled_tensor = torch.tensor(layer_scaled, device=self.device, dtype=torch.float32)
            time_coords = torch.sqrt(1 + torch.sum(layer_scaled_tensor**2, dim=1, keepdim=True))
            lorentz_activations[:, l, :] = torch.cat([time_coords, layer_scaled_tensor], dim=1)
            
        self.background_activations_by_layer = lorentz_activations
        # ================================================================
        
        self.background_labels = all_labels
        
        if os.path.exists(f"./{self.your_flag}/training_sequences.pt"):
            training_sequences = torch.load(f"./{self.your_flag}/training_sequences.pt")
        else:
            training_sequences = self._get_training_sequences()
            
        y_train = self.background_labels
        benign_indices = torch.where(y_train == 0)[0]
        benign_training_sequences = training_sequences[benign_indices]
        self.mean = benign_training_sequences.mean(dim=0, keepdim=True)
        self.std = benign_training_sequences.std(dim=0, keepdim=True) + 1e-8
        X_train = (benign_training_sequences - self.mean) / self.std
        self.if_model = PyTorchIsolationForest(n_estimators=n_estimators, max_samples=512, random_state=42)
        self.if_model.fit(X_train)

    def predict(self, prompt_text: str = None, input_ids: torch.Tensor = None, return_score=True, attack_key=None,
                    return_ranks=False):
            if input_ids is None and prompt_text is not None:
                messages = [{"role": "user", "content": prompt_text}]
                input_ids = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False
                ).to(self.device)
            elif input_ids is not None:
                input_ids = input_ids.to(self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                if prompt_text is None:
                    prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            else:
                raise ValueError("Either prompt_text or input_ids must be provided!")

            new_activations = self.get_last_token_hidden_states(input_ids)

            # === HYPERBOLIC MTK CHANGE 2: Project Test Prompt ===
            # Apply the same PCA transformation and Lorentz scaling to the test prompt
            new_activations_lorentz = torch.empty((self.num_layers, 64 + 1), device=self.device)
            for l in range(self.num_layers):
                layer_data = new_activations[l].unsqueeze(0).cpu().numpy()
                layer_pca = self.pca_models[l].transform(layer_data)
                layer_scaled = layer_pca * 15.0
                layer_scaled_tensor = torch.tensor(layer_scaled, device=self.device, dtype=torch.float32)
                time_coord = torch.sqrt(1 + torch.sum(layer_scaled_tensor**2, dim=1, keepdim=True))
                new_activations_lorentz[l] = torch.cat([time_coord, layer_scaled_tensor], dim=1).squeeze(0)
            
            new_activations = new_activations_lorentz
            # =====================================================

            ranks = self._calculate_single_rank_k_nb(
                new_activations,
                self.background_activations_by_layer,
                0,
                self.background_labels,
                k=self.k_nb,
                device=self.device
            )

            scaled_sequence = (ranks - self.mean) / self.std

            anomaly_score = self.if_model.decision_function(scaled_sequence)[0].item()
            if anomaly_score < 0:
                label_str = "Jailbreak Prompt"
                pred_label = 1
            else:
                label_str = "Benign prompt"
                pred_label = 0

            result = [label_str, pred_label]

            if return_score:
                result.append(anomaly_score)

            if return_ranks:
                result.append(ranks.cpu().numpy())

            return tuple(result) if len(result) > 1 else result[0]

    def _restructure_activations(self, activations_list):
        if not activations_list:
            return []
        num_layers = len(activations_list[0])
        activations_by_layer = [[] for _ in range(num_layers)]
        for sample_activations in activations_list:
            for i in range(num_layers):
                activations_by_layer[i].append(sample_activations[i])
        return [torch.stack(layer_acts, dim=0) for layer_acts in activations_by_layer]

    def _get_training_sequences(self):
        num_samples = len(self.background_labels)
        num_layers = self.num_layers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_sequences = torch.empty((num_samples, num_layers), device=device)
        background_activations_gpu = self.background_activations_by_layer
        background_labels_gpu = self.background_labels

        for i in tqdm(range(num_samples), desc="Generating training sequences"):
            current_vector = background_activations_gpu[i]
            mask = torch.ones(num_samples, dtype=torch.bool, device=device)
            mask[i] = False
            other_vectors = background_activations_gpu[mask]
            other_labels = background_labels_gpu[mask]

            ranks = self._calculate_single_rank_k_nb(
                current_vector,
                other_vectors,
                0,
                other_labels,
                k=self.k_nb,
                device=device
            )
            all_sequences[i] = ranks
            
        save_dir = os.path.dirname(f"./{self.your_flag}/training_sequences.pt")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(all_sequences, f"./{self.your_flag}/training_sequences.pt")
        return all_sequences

    def _calculate_single_rank_k_nb(self, test_vector, background_vectors, target_label, background_labels_arr, k,
                                    device):
        test_vector = test_vector.unsqueeze(0)

        # === HYPERBOLIC MTK CHANGE 3: Minkowski Distance instead of Euclidean ===
        # The original code calculated Euclidean distance using `.norm(p=2)`.
        # Because we mapped to Lorentz space, we must calculate the Minkowski Inner Product,
        # followed by the arcosh to get the true Hyperbolic distance.
        
        # Time coordinate is at index 0, Space coordinates at index 1:
        u0 = background_vectors[:, :, 0]
        v0 = test_vector[:, :, 0]
        u_space = background_vectors[:, :, 1:]
        v_space = test_vector[:, :, 1:]
        
        # Minkowski product: -u0*v0 + sum(u_i * v_i)
        minkowski_product = -(u0 * v0) + torch.sum(u_space * v_space, dim=2)
        
        # Clamp to -1.0 to avoid NaNs (arcosh of values slightly > -1 due to float math can crash)
        minkowski_product = torch.clamp(minkowski_product, max=-1.0)
        
        # Lorentz Distance formula
        layer_distances = torch.acosh(-minkowski_product).permute(1, 0)
        # ========================================================================

        sorted_indices = torch.argsort(layer_distances, dim=1)
        sorted_background_labels = background_labels_arr[sorted_indices]
        match_indices_in_sorted_tensor = torch.empty((self.num_layers), device=device)
        for i, s in enumerate(sorted_background_labels):
            match_indices_in_sorted_tensor[i] = (torch.where(s == target_label)[0] + 1)[:k].float().mean()
        return match_indices_in_sorted_tensor

    def get_output(self, input_ids, max_new_tokens=100):
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens
            ).sequences
        new_tokens = output_ids[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def get_last_token_hidden_states(self, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long().to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states = outputs.hidden_states[1:]
        last_token_states = torch.stack([h[:, -1, :] for h in hidden_states], dim=1).squeeze(0)

        return last_token_states