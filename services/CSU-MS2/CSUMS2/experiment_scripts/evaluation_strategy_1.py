# allow to import modules from the project root directory
import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import bisect
from infer import ModelInference
import numpy as np
import torch
from tqdm import tqdm
from matchms.importing import load_from_mgf
import pandas as pd
import os
import matchms.filtering as msfilters
import glob
from functools import lru_cache
import gc
import time

# Enable memory efficient processing
torch.backends.cudnn.benchmark = True

def spectrum_processing(s):
    """This is how one would typically design a desired pre- and post-
    processing pipeline."""
    s = msfilters.normalize_intensities(s)
    s = msfilters.select_by_mz(s, mz_from=0, mz_to=1500)
    return s

@lru_cache(maxsize=1000)
def calculate_weights(collision_energy):
    """Cache weight calculations for repeated collision energies"""
    eps = 1e-10
    inv_low = 1 / (abs(collision_energy - 10) + eps)
    inv_med = 1 / (abs(collision_energy - 20) + eps)
    inv_high = 1 / (abs(collision_energy - 40) + eps)
    
    total = inv_low + inv_med + inv_high
    return inv_low / total, inv_med / total, inv_high / total

class ModelInferenceManager:
    """Manages multiple models efficiently"""
    def __init__(self, device="cpu"):
        self.device = device
        
        # Initialize models with memory optimization
        config_paths = {
            'low': "model/qtof_model/low_energy/checkpoints/config.yaml",
            'median': "model/qtof_model/median_energy/checkpoints/config.yaml", 
            'high': "model/qtof_model/high_energy/checkpoints/config.yaml"
        }
        
        model_paths = {
            'low': "model/qtof_model/low_energy/checkpoints/model.pth",
            'median': "model/qtof_model/median_energy/checkpoints/model.pth",
            'high': "model/qtof_model/high_energy/checkpoints/model.pth"
        }
        
        self.models = {}
        for energy in ['low', 'median', 'high']:
            self.models[energy] = ModelInference(
                config_path=config_paths[energy],
                pretrain_model_path=model_paths[energy],
                device=device
            )
            
    def encode_spectrum_all_energies(self, spectrum):
        """Encode spectrum with all three energy models at once"""
        with torch.no_grad():
            features = {}
            for energy in ['low', 'median', 'high']:
                features[energy] = self.models[energy].ms2_encode([spectrum])
                # Ensure tensor is on the correct device
                if hasattr(features[energy], 'to'):
                    features[energy] = features[energy].to(self.device)
        return features
    
    def encode_smiles_all_energies(self, smiles_list):
        """Encode SMILES with all three energy models at once"""
        with torch.no_grad():
            features = {}
            for energy in ['low', 'median', 'high']:
                # Get SMILES features and keep them on the same device as models
                smiles_features = self.models[energy].smiles_encode(smiles_list)
                if hasattr(smiles_features, 'to'):
                    features[energy] = smiles_features.to(self.device)
                else:
                    features[energy] = smiles_features
        return features

def process_single_spectrum(mgf_file, file_idx, model_manager):
    """Process a single spectrum file efficiently"""
    try:
        # Load and process spectrum
        spectrum = list(load_from_mgf(mgf_file))[0]
        spectrum = spectrum_processing(spectrum)
        
        # Get spectrum features for all energies
        ms_features = model_manager.encode_spectrum_all_energies(spectrum)
        
        # Extract metadata
        query_ms = float(spectrum.metadata['precursor_mz']) - 1.008
        collision_energy = int(spectrum.metadata.get('collision_energy', 20))
        smiles_lst = [spectrum.metadata['smiles']]
        
        # Get SMILES features for all energies
        smiles_features = model_manager.encode_smiles_all_energies(smiles_lst)
        
        # Calculate similarities efficiently - ensure all tensors are on same device
        similarities = {}
        for energy in ['low', 'median', 'high']:
            # Make sure both tensors are on the same device
            ms_feat = ms_features[energy]
            smiles_feat = smiles_features[energy]
            
            # Ensure both are on the same device
            if hasattr(ms_feat, 'device') and hasattr(smiles_feat, 'device'):
                if ms_feat.device != smiles_feat.device:
                    smiles_feat = smiles_feat.to(ms_feat.device)
            
            sim = ms_feat @ smiles_feat.t()
            similarities[energy] = sim.cpu().numpy()  # Move to CPU for numpy operations
        
        # Calculate weights once
        weight1, weight2, weight3 = calculate_weights(collision_energy)
        
        # Compute weighted similarity
        weighted_similarity = (weight1 * similarities['low'] + 
                             weight2 * similarities['median'] + 
                             weight3 * similarities['high'])
        
        weighted_similarity = np.squeeze(weighted_similarity, axis=0)
        
        # Create results efficiently
        scores = weighted_similarity.tolist() if hasattr(weighted_similarity, 'tolist') else [weighted_similarity]
        
        result_data = {
            'spectrum_id': file_idx,
            'mgf_file': os.path.basename(mgf_file),
            'SMILES': smiles_lst[0],  # Single SMILES
            'Score': scores[0],       # Single score
            'Rank': 1,               # Single rank
            'collision_energy': collision_energy,
            'precursor_mz': query_ms + 1.008
        }
        
        # Clean up memory
        del ms_features, smiles_features, similarities
        if model_manager.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return result_data
        
    except Exception as e:
        print(f"Error processing file {mgf_file}: {e}")
        return None

def main():
    # Setup
    eval_dir = "/home/i_golov/csmp_search_engine_for_specmol/CSMP_thesis_project/data/eval_1"
    mgf_files = glob.glob(os.path.join(eval_dir, "*.mgf"))
    print(f"Found {len(mgf_files)} MGF files for evaluation")
    
    # Initialize model manager
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model_manager = ModelInferenceManager(device=device)
    
    # Process files efficiently
    all_results = []
    temp_save_dir = "eval_1_results"
    os.makedirs(temp_save_dir, exist_ok=True)
    
    batch_size = 1024  # Process in batches for memory efficiency
    total_batches = (len(mgf_files) + batch_size - 1) // batch_size
    
    # Single progress bar for all files
    successful_count = 0
    failed_count = 0
    
    with tqdm(total=len(mgf_files), 
              desc="Processing MGF files", 
              unit="files",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}') as pbar:
        
        for batch_start in range(0, len(mgf_files), batch_size):
            batch_end = min(batch_start + batch_size, len(mgf_files))
            batch_files = mgf_files[batch_start:batch_end]
            batch_num = batch_start // batch_size + 1
            
            batch_results = []
            batch_start_time = time.time()
            
            for i, mgf_file in enumerate(batch_files):
                file_idx = batch_start + i
                
                # Update progress bar with current file info
                current_file = os.path.basename(mgf_file)
                if len(current_file) > 25:
                    current_file = current_file[:22] + "..."
                
                pbar.set_postfix({
                    'batch': f"{batch_num}/{total_batches}",
                    'current': current_file,
                    'success': successful_count,
                    'failed': failed_count
                })
                
                result = process_single_spectrum(mgf_file, file_idx, model_manager)
                
                if result is not None:
                    batch_results.append(result)
                    successful_count += 1
                else:
                    failed_count += 1
                
                # Update progress
                pbar.update(1)
            
            # Convert batch to DataFrame and save
            if batch_results:
                batch_df = pd.DataFrame(batch_results)
                all_results.append(batch_df)
                
                # Save intermediate results
                temp_filename = f"{temp_save_dir}/results_batch_{batch_num:03d}.csv"
                batch_df.to_csv(temp_filename, index=False)
                
                batch_time = time.time() - batch_start_time
                pbar.write(f"✓ Batch {batch_num}/{total_batches} completed: {len(batch_results)}/{len(batch_files)} successful in {batch_time:.1f}s")
            else:
                pbar.write(f"✗ Batch {batch_num}/{total_batches} failed: 0/{len(batch_files)} successful")
                
            # Clean up memory after each batch
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Save final results
    if all_results:
        final_results = pd.concat(all_results, ignore_index=True)
        final_results.to_csv("final_spectrum_analysis_results.csv", index=False)
        print(f"\n✓ Saved final results to final_spectrum_analysis_results.csv")
        print(f"📊 Total processed: {successful_count}/{len(mgf_files)} successful ({successful_count/len(mgf_files)*100:.1f}%)")
        print(f"❌ Failed: {failed_count} files")
    else:
        print("\n✗ No results to save - all processing failed")

if __name__ == "__main__":
    main()