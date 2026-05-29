from mp_api.client import MPRester
from pymatgen.electronic_structure.bandstructure import BandStructure
from pymatgen.electronic_structure.core import Spin
import pandas as pd
import numpy as np
from emmet.core.electronic_structure import BSPathType

# Set your API key (get it from https://materialsproject.org/dashboard)

with open("./api_key", 'r') as f:
    API_KEY=f.readline()[:-1]
    print(len(API_KEY))
    pass


def extract_band_structure_data(material_id, path_type=BSPathType.setyawan_curtarolo):
    """
    Extract band structure data for a given material ID and convert to tabular format
    """
    with MPRester(API_KEY) as mpr:
        try:
            # Get the band structure object
            bs = mpr.get_bandstructure_by_material_id(material_id, path_type=path_type)
            
            if bs is None:
                print(f"No band structure data available for {material_id}")
                return None
            
            # Extract data into a list of dictionaries
            data_rows = []
            
            # Get k-point labels and coordinates
            kpoints = bs.kpoints
            labels_dict = bs.labels_dict
            
            # Iterate through each band and k-point
            for band_idx, band in enumerate(bs.bands[Spin.up]):
                for k_idx, (kpoint, energy) in enumerate(zip(kpoints, band)):
                    # Determine if this k-point has a label
                    k_label = None
                    for label, label_kpoint in labels_dict.items():
                        if np.allclose(kpoint.frac_coords, label_kpoint.frac_coords, atol=1e-3):
                            k_label = label
                            break
                    
                    row = {
                        'material_id': material_id,
                        'band_index': band_idx,
                        'kpoint_index': k_idx,
                        'kpoint_frac_coords_x': kpoint.frac_coords[0],
                        'kpoint_frac_coords_y': kpoint.frac_coords[1],
                        'kpoint_frac_coords_z': kpoint.frac_coords[2],
                        'kpoint_label': k_label,
                        'energy_eV': energy,
                        'efermi': bs.efermi,
                        'band_gap': bs.get_band_gap()['energy'] if bs.get_band_gap() else None,
                        'is_metal': bs.is_metal()
                    }
                    
                    # Add spin-down data if available (for magnetic materials)
                    if Spin.down in bs.bands:
                        row['energy_spin_down_eV'] = bs.bands[Spin.down][band_idx][k_idx]
                    else:
                        row['energy_spin_down_eV'] = None
                    
                    data_rows.append(row)
            
            return pd.DataFrame(data_rows)
            
        except Exception as e:
            print(f"Error processing {material_id}: {str(e)}")
            return None

def get_band_structure_summary(material_ids):
    """
    Get a summary of band structure properties for multiple materials
    """
    with MPRester(API_KEY) as mpr:
        summary_data = []
        
        for mid in material_ids:
            try:
                # Get summary document with electronic structure info
                docs = mpr.materials.summary.search(
                    material_ids=[mid], 
                    fields=["material_id", "formula_pretty", "band_gap", 
                           "is_metal", "cbm", "vbm", "efermi", "is_gap_direct"]
                )
                
                if docs:
                    doc = docs[0]
                    summary_data.append({
                        'material_id': mid,
                        'formula': doc.formula_pretty,
                        'band_gap_eV': doc.band_gap,
                        'is_metal': doc.is_metal,
                        'cbm_eV': doc.cbm,
                        'vbm_eV': doc.vbm,
                        'efermi_eV': doc.efermi,
                        'is_gap_direct': doc.is_gap_direct
                    })
            except Exception as e:
                print(f"Error getting summary for {mid}: {str(e)}")
        
        return pd.DataFrame(summary_data)

# Example Usage:

# 1. Define material IDs to query
material_ids = ["mp-149", "mp-19770", "mp-1101025"]  # Si, GaAs, etc.

# 2. Get detailed band structure data (this creates large files)
print("Fetching detailed band structure data...")
all_band_data = []

for mid in material_ids:
    df = extract_band_structure_data(mid)
    if df is not None:
        all_band_data.append(df)

if all_band_data:
    combined_df = pd.concat(all_band_data, ignore_index=True)
    combined_df.to_csv('band_structures_detailed.csv', index=False)
    print(f"Saved detailed band structure data to 'band_structures_detailed.csv'")
    print(f"Total rows: {len(combined_df)}")

# 3. Get summary data (smaller file with key properties)
print("\nFetching band structure summary...")
summary_df = get_band_structure_summary(material_ids)
summary_df.to_csv('band_structures_summary.csv', index=False)
print(f"Saved summary to 'band_structures_summary.csv'")
print(summary_df)

# 4. Alternative: Query by criteria (e.g., all Si-O compounds with band gaps > 2eV)
print("\nQuerying materials by criteria...")
with MPRester(API_KEY) as mpr:
    # Find materials with band structure data
    docs = mpr.materials.summary.search(
        chemsys="Si-O",
        band_gap=(2, None),
        fields=["material_id", "formula_pretty", "band_gap", "bandstructure"]
    )
    
    criteria_data = []
    for doc in docs[:10]:  # Limit to first 10
        # Check if bandstructure field exists (indicates data availability)
        has_bs = doc.bandstructure is not None if hasattr(doc, 'bandstructure') else False
        criteria_data.append({
            'material_id': doc.material_id,
            'formula': doc.formula_pretty,
            'band_gap_eV': doc.band_gap,
            'has_bandstructure_data': has_bs
        })
    
    criteria_df = pd.DataFrame(criteria_data)
    criteria_df.to_csv('materials_by_criteria.csv', index=False)
    print(f"Found {len(criteria_df)} materials matching criteria")
    print(criteria_df.head())