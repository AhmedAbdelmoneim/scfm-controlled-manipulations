import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Datasets for evaluation
    """)
    return


@app.cell
def _():
    import os
    import scanpy as sc

    return os, sc


@app.cell
def _():
    ATLASES_DIR = "/vault/amoneim/atlases/"
    OUTPUT_DIR = "/vault/amoneim/scfm-controlled-manipulations/raw_datasets"

    ATLASES = {
        "arterial": "human_arterial_cell_atlas.h5ad",
        "immune": "human_immune_health_atlas.h5ad",
        "retina": "human_retina_cell_atlas.h5ad",
        "tabula_sapiens": "tabula_sapiens.h5ad",
        "lung": "human_lung_cell_atlas.h5ad",
        "brain": "human_brain_cell_atlas.h5ad",
    }
    return ATLASES, ATLASES_DIR, OUTPUT_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Subsample
    """)
    return


@app.cell
def _(ATLASES, ATLASES_DIR, OUTPUT_DIR, os, sc):
    def subsample_and_save_atlases(total_cells=20000, random_state=42):
        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        for key, filename in ATLASES.items():
            in_path = os.path.join(ATLASES_DIR, filename)
            out_path = os.path.join(OUTPUT_DIR, f"{key}.h5ad")

            print(f"Processing '{key}' atlas...")
            print(f"  Reading: {in_path}")

            try:
                # 1. Load in read-only backed mode
                adata = sc.read_h5ad(in_path, backed="r")

                # 2. Perform memory-efficient random sampling on .obs
                obs_df = adata.obs
            
                # Prevent sampling errors if the atlas has fewer cells than requested
                n_sample = min(total_cells, len(obs_df))
            
                # Sample directly across the entire dataframe
                sampled_indices = obs_df.sample(n=n_sample, random_state=random_state).index

                print(
                    f"  Subsampling {len(sampled_indices)} cells from a total of {len(obs_df)}."
                )
                print(f"  Writing to: {out_path}")

                # 3. Slice and write the subset directly to disk
                adata[sampled_indices].copy(filename=out_path)

                # Clean up the backed file lock before moving to the next one
                adata.file.close()
                print("  Success.\n")

            except Exception as e:
                print(f"  Error processing '{key}': {e}\n")

    subsample_and_save_atlases()
    return


if __name__ == "__main__":
    app.run()
