#!/bin/bash
# Rename coarse_* dimension/meta variables → cond_* in notebooks
# Does NOT rename coarse_model, coarse_emb, self.coarse, gen_coarse_*, etc.
NBS=/workspaces/ClaudeCode-Mar12/midi-rae/nbs

sed -i 's/coarse_level_dims/cond_level_dims/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"
sed -i 's/coarse_level_names/cond_level_names/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"
sed -i 's/coarse_source_scales/cond_source_scales/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"
sed -i 's/coarse_n_comp/cond_n_comp/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"
sed -i 's/coarse_n_patches/cond_n_patches/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"
sed -i 's/coarse_dim/cond_dim/g' "$NBS/01_data.ipynb" "$NBS/12_train_flow.ipynb"

echo "Done."
