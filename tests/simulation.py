import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from scipy.stats import zscore

def simulate_gwas_data(n_snps=10000, n_samples=10000):
    """Simulate GWAS summary statistics"""
    np.random.seed(42)
    return pd.DataFrame({
        'SNP': [f'rs{np.random.randint(1e6,1e9)}' for _ in range(n_snps)],
        'CHR': np.random.choice(range(1,23), n_snps),
        'POS': np.random.randint(1e6, 1e9, n_snps),
        'BETA': np.concatenate([
            np.random.normal(0, 0.1, n_snps-100),
            np.random.normal(0.3, 0.15, 100)  
        ]),
        'P': np.concatenate([
            np.random.uniform(0, 1, n_snps-100),
            np.random.uniform(0, 0.0001, 100)  
        ])
    })

def load_pathway_annotations():
    """Simulate pathway annotations (Replace with KEGG/Reactome)"""
    return {
        'amyloid_processing': ['rs12345', 'rs23456', 'rs34567'],
        'immune_response': ['rs45678', 'rs56789', 'rs67890'],
        'lipid_metabolism': ['rs78901', 'rs89012', 'rs90123']
    }


class AD_SHARP:
    def __init__(self, pathway_annotations):
        self.pathways = pathway_annotations
        self.selected_snps = []
        self.snp_weights = {}
        
    def lasso_selection(self, gwas_data, alpha=0.01):
        """Step 1: LASSO-based SNP selection"""
        X = np.abs(gwas_data['BETA']).values.reshape(-1, 1)
        y = -np.log10(gwas_data['P'])
        
        model = LassoCV(alphas=[alpha], max_iter=10000, cv=3)
        model.fit(X, y)
        
        selected_idx = np.where(model.coef_ != 0)[0]
        self.selected_snps = gwas_data.iloc[selected_idx]['SNP'].tolist()
        return self.selected_snps
    
    def pathway_weighting(self, gwas_data):
        """Step 2: Assign pathway-based weights"""
        for snp in self.selected_snps:
            base_weight = gwas_data.loc[gwas_data['SNP'] == snp, 'BETA'].values[0]
            
            # Check pathway membership
            pathway_boost = 1.0
            for pathway, snps in self.pathways.items():
                if snp in snps:
                    pathway_boost *= 1.5  # Weight boost for pathway SNPs
            
            self.snp_weights[snp] = base_weight * pathway_boost
        return self.snp_weights
    
    def calculate_prs(self, genotype_df):
        """Step 3: Compute weighted PRS"""
        prs_scores = []
        for _, row in genotype_df.iterrows():
            score = 0
            for snp, weight in self.snp_weights.items():
                if snp in genotype_df.columns:
                    score += row[snp] * weight
            prs_scores.append(score)
        
        return zscore(prs_scores)  # Standardized scores


if __name__ == "__main__":
    print("Building AD-SHARP prototype...")
    
    # 1. Load data (simulated)
    gwas_data = simulate_gwas_data()
    pathways = load_pathway_annotations()
    
    # 2. Initialize AD-SHARP
    tool = AD_SHARP(pathways)
    
    # 3. SNP selection
    selected = tool.lasso_selection(gwas_data)
    print(f"Selected {len(selected)} SNPs via LASSO")
    
    # 4. Pathway weighting
    weights = tool.pathway_weighting(gwas_data)
    
    # 5. Simulate genotype data (individuals x SNPs)
    genotype_sim = pd.DataFrame(
        np.random.binomial(2, 0.3, (100, len(selected))),
        columns=selected
    )
    
    # 6. Calculate PRS
    prs_scores = tool.calculate_prs(genotype_sim)
    print(f"Computed PRS for {len(prs_scores)} individuals")
    print(f"PRS range: {min(prs_scores):.2f} to {max(prs_scores):.2f} (SD units)")
    
    print("\nAD-SHARP prototype ready! Next steps:")
    print("- Replace simulated data with IGAP GWAS (https://www.niagads.org/adsp)")
    print("- Add GTEx brain eQTLs (https://gtexportal.org)")
    print("- Validate with ADNI genotypes (https://adni.loni.usc.edu)")