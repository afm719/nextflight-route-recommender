import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import scipy.sparse as sparse
import csv
import os
from collections import defaultdict


class CustomWMF:

    """Simple Weighted Matrix Factorization helper used by the API.

    This class exposes a convenience method to compute a latent vector for a
    new 'folded-in' profile built from a small set of liked item indices and
    to return top-N recommendations based on cosine/inner-product scores.

    The method implements the standard WMF per-row update used in ALS for
    implicit feedback, solving for x_u in the linear system:

        (Y^T C_u Y + λ I) x_u = Y^T C_u p(u)

    Here C_u is a diagonal confidence matrix for the new profile (C = 1 + αR)
    and p(u) is the binary preference vector (1 for liked items).
    """

    def __init__(self, factors=50, regularization=0.1, iterations=15, alpha=15):
        # Model hyperparameters (kept for compatibility with pickled objects)
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha

        # Latent factor matrices. They are expected to be populated by the
        # deserialized model object (loaded from disk). Types:
        #   user_vectors: ndarray, shape (n_users, k)
        #   item_vectors: ndarray, shape (n_items, k)
        self.user_vectors = None
        self.item_vectors = None

    def recommend_for_new_profile(self, liked_item_indices, N=5):
        """Return top-N recommendations for a new profile built from liked items.

        Parameters
        ----------
        liked_item_indices : list[int]
            Indices of items that the new profile 'likes' (observed items).
        N : int
            Number of recommendations to return.

        Returns
        -------
        (indices, scores)
            Tuple with recommended item indices and the corresponding scores.
        """
        Y = self.item_vectors

        # Precompute Y^T Y which is reused in the per-profile linear system
        YtY = Y.T.dot(Y)

        # Regularization matrix λ I in sparse CSR format to match shapes
        lambda_I = sparse.eye(self.factors).tocsr() * self.regularization

        # Confidence values for the liked items: here we use alpha for all liked items
        conf = np.ones(len(liked_item_indices)) * self.alpha

        # Subset of item factors corresponding to the liked items
        Y_sub = Y[liked_item_indices]

        # Left-hand side: Y^T Y + Y_sub^T * diag(conf) * Y_sub + λ I
        A = YtY + Y_sub.T.dot(sparse.diags(conf).dot(Y_sub)) + lambda_I

        # Right-hand side: Y_sub^T * (conf + 1)
        # Explanation: since C = 1 + α R and p(u)=1 for observed items,
        # Y^T C_u p(u) reduces to Y_sub^T * (conf + 1)
        b = Y_sub.T.dot(conf + 1)

        # Solve the small k×k system to obtain the new latent vector
        new_vec = np.linalg.solve(A, b)

        # Compute scores and exclude the already liked items
        scores = self.item_vectors.dot(new_vec)
        scores[liked_item_indices] = -9999

        # Return indices of top-N scores (descending)
        idx = np.argsort(scores)[::-1][:N]
        return idx, scores[idx]


import __main__

__main__.CustomWMF = CustomWMF
sys.modules['__main__'].CustomWMF = CustomWMF



# ==================== JACCARD SIMILARITY ====================

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    
    Parameters
    ----------
    set_a : set
        First set of tourism types.
    set_b : set
        Second set of tourism types.
        
    Returns
    -------
    float
        Similarity score between 0 and 1.
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ==================== LOAD MODEL AND METADATA ====================

# Load the serialized WMF model and the airport index mappings
#this model was trained on the full dataset with 100 factors, 0.1 regularization, 15 iterations and alpha=15, and was
#trained in the notebook ./model/recommendation_model_training.ipynb 
with open('./model/wmf_model.npz', 'rb') as f:
    wmf_model = pickle.load(f)
    
with open('./model/mappings.pkl', 'rb') as f:
    m = pickle.load(f)
    airport_map, idx_map = m['airport_to_idx'], m['idx_to_airport']
   
iata_to_city: dict = {}
iata_to_coord: dict = {}
iata_to_tourism: dict = {}  # Map IATA code to set of tourism types

airports_dat = os.path.join(os.path.dirname(__file__), 'data', 'airports.dat')
if os.path.exists(airports_dat):
    try:
        with open(airports_dat, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if len(row) > 7:
                    iata = row[4].strip().upper()
                    city = row[2].strip()
                    name = row[1].strip()
                    lat = row[6].strip()
                    lon = row[7].strip()
                    if iata and iata != '\\N':
                        # prefer city name, fallback to airport name
                        iata_to_city[iata] = city or name or iata
                        try:
                            iata_to_coord[iata] = [float(lat), float(lon)]
                        except Exception:
                            # ignore parse errors, leave coord absent
                            pass
    except Exception as e:
        print(f"Warning: Could not load airports.dat: {e}")
else:
    print(f"Warning: airports.dat not found at {airports_dat}")

# Load tourism attributes for airports
airport_tourism_csv = os.path.join(os.path.dirname(__file__), 'data', 'airport_tourism.csv')
if os.path.exists(airport_tourism_csv):
    try:
        with open(airport_tourism_csv, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            tourism_types = ['beach', 'mountain', 'culture', 'business', 'adventure', 'nightlife', 'food', 'nature', 'budget', 'luxury']
            for row in reader:
                iata = row['iata'].strip().upper()
                # Build set of tourism types where value is "1"
                tourism_set = set()
                for tourism_type in tourism_types:
                    if row.get(tourism_type, '0').strip() == '1':
                        tourism_set.add(tourism_type)
                iata_to_tourism[iata] = tourism_set
    except Exception as e:
        print(f"Warning: Could not load airport_tourism.csv: {e}")
else:
    print(f"Warning: airport_tourism.csv not found at {airport_tourism_csv}")


app = FastAPI()

# Allow cross-origin requests from any origin for the simple web demo.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Selection(BaseModel):
    """Request schema for /recommend endpoint.

    airports: list of IATA codes (strings) that represent the selected items.
    """
    airports: list[str]


@app.post("/recommend")
async def recommend(selection: Selection):
    """Return top-N recommended airports for a folded-in profile.

    The endpoint maps incoming IATA codes to internal indices, constructs a
    folded-in profile via `wmf_model.recommend_for_new_profile` and formats the
    response as a JSON list with IATA codes and scores.
    """
    # Map provided IATA codes to internal indices, ignoring unknown codes
    indices = [airport_map[iata] for iata in selection.airports if iata in airport_map]
    if not indices:
        return {"recommendations": []}

    recs_idx, recs_scr = wmf_model.recommend_for_new_profile(indices, N=5)
    return {
        "recommendations": [
            {
                "iata": idx_map[i], 
                "city": iata_to_city.get(idx_map[i]) or idx_map[i],  # Use IATA as fallback if no city
                "score": round(float(s), 4)
            }
            for i, s in zip(recs_idx, recs_scr)
        ]
    }


@app.get("/airports")
async def get_airports():
    """Return a lightweight catalog of available airports.

    The endpoint returns a list of simple objects with `iata` and `city` keys
    to be consumed by the front-end. For now, the city field mirrors the IATA
    code; the mapping can be extended with richer metadata later.
    """
    # Return city metadata when available to allow city->IATA lookups on frontend
    # Include coordinates when available so the frontend can render maps client-side
    result = []
    for iata in airport_map.keys():
        coord = iata_to_coord.get(iata)
        lat = coord[0] if coord else None
        lon = coord[1] if coord else None
        tourism_types = list(iata_to_tourism.get(iata, set()))
        result.append({
            "iata": iata,
            "city": iata_to_city.get(iata, iata),
            "lat": lat,
            "lon": lon,
            "tourism": tourism_types
        })
    return result


class SelectionWithTourism(BaseModel):
    """Request schema for /recommend-with-tourism endpoint.
    
    airports: list of IATA codes (strings) that the user has visited.
    tourism_preferences: list of tourism type names (["beach", "culture", "nightlife"]).
    """
    airports: list[str]
    tourism_preferences: list[str]


@app.post("/recommend-with-tourism")
async def recommend_with_tourism(selection: SelectionWithTourism):
    """Return top-N recommended airports based on WMF + Jaccard similarity.
    
    This endpoint combines two recommendation signals:
    1. Weighted Matrix Factorization (WMF) scores based on visited airports
    2. Jaccard similarity between user's tourism preferences and airport tourism profiles
    
    The final score is a weighted blend of both signals.
    
    Parameters
    ----------
    selection : SelectionWithTourism
        User's visited airports and desired tourism types.
        
    Returns
    -------
    dict
        JSON with list of recommendations including IATA, WMF score, Jaccard score, and final score.
    """
    # Map provided IATA codes to internal indices, ignoring unknown codes
    indices = [airport_map[iata] for iata in selection.airports if iata in airport_map]
    if not indices:
        return {"recommendations": []}

    # Get WMF-based recommendations
    recs_idx, recs_scr_wmf = wmf_model.recommend_for_new_profile(indices, N=20)  # Get more to filter by tourism
    
    # User's tourism preference set
    user_tourism_set = set(selection.tourism_preferences) if selection.tourism_preferences else set()
    
    # Calculate combined scores
    recommendations = []
    for i, wmf_score in zip(recs_idx, recs_scr_wmf):
        iata = idx_map[i]
        airport_tourism_set = iata_to_tourism.get(iata, set())
        
        # Jaccard similarity based on tourism preferences
        jaccard_score = jaccard_similarity(user_tourism_set, airport_tourism_set)
        
        # Blend both scores (60% WMF, 40% Jaccard)
        # Normalize WMF score to [0, 1] range for fair blending
        wmf_normalized = float(wmf_score) / (float(wmf_score) + 1e-6) if wmf_score > 0 else 0
        wmf_normalized = min(1.0, wmf_normalized)  # Cap at 1
        
        final_score = 0.6 * wmf_normalized + 0.4 * jaccard_score
        
        recommendations.append({
            "iata": iata,
            "city": iata_to_city.get(iata) or iata,  # Use IATA as fallback if no city
            "wmf_score": round(float(wmf_score), 4),
            "jaccard_score": round(jaccard_score, 4),
            "final_score": round(final_score, 4),
            "tourism_types": list(airport_tourism_set)
        })
    
    # Sort by final score and return top 5
    recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)[:5]
    
    return {"recommendations": recommendations}


@app.get("/tourism-types")
async def get_tourism_types():
    """Return available tourism types for filtering and selection.
    
    Returns
    -------
    dict
        JSON with list of all available tourism type names.
    """
    tourism_types = ['beach', 'mountain', 'culture', 'business', 'adventure', 'nightlife', 'food', 'nature', 'budget', 'luxury']
    return {"tourism_types": tourism_types}