# Next Trip Route Recommender

A flight route recommendation system leveraging **Weighted Matrix Factorization (WMF)** with collaborative filtering and semantic similarity matching. This project combines latent factor models with tourism preference analysis to deliver personalized airport recommendations based on user travel history.

**How to use it**: 
 1. ``` git clone https://github.com/afm719/nextflight-route-recommender ```
 2. ``` sudo apt-get install python3-dev ```  (because the code has some C/C++ dependencies)
 3. ``` bash RUN_PROJECT_MAC_LINUX.sh```


## Project Overview

This application implements an advanced recommendation engine that analyzes user travel patterns and preferences to suggest the next ideal destination. By integrating collaborative filtering through matrix factorization with Jaccard similarity-based tourism classification, the system provides dual-signal recommendations that balance both user behavior patterns and explicit travel interests.

The system is built with:
- **Backend**: FastAPI (async Python web framework)
- **Frontend**: Vanilla JavaScript + HTML5 (no build tools required)
- **Algorithm**: Alternating Least Squares (ALS) for Implicit Feedback Recommendation
- **Enhancement**: Hybrid scoring with tourism preference matching

## Mathematical Foundation

### 1. Weighted Matrix Factorization (WMF)

The core recommendation algorithm is built on **Alternating Least Squares (ALS)** for implicit feedback, a state-of-the-art technique in collaborative filtering.

#### Problem Formulation

Given an implicit feedback matrix $\mathbf{R} \in \mathbb{R}^{m \times n}$ where:
- $m$ represents the number of users (each user represents a unique airport visit pattern)
- $n$ represents the number of airports
- $R_{ij} = 1$ if user $i$ has visited airport $j$, and $0$ otherwise

We factorize the matrix as:
$$\mathbf{R} \approx \mathbf{U} \mathbf{V}^T$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times k}$ is the user latent factor matrix
- $\mathbf{V} \in \mathbb{R}^{n \times k}$ is the item (airport) latent factor matrix
- $k$ is the number of latent factors (set to 50 in this implementation)

#### Confidence-Weighted Loss Function

The objective function for WMF with implicit feedback is:
$$\mathcal{L} = \sum_{i,j} C_{ij}(R_{ij} - \mathbf{u}_i^T \mathbf{v}_j)^2 + \lambda(\|\mathbf{U}\|^2_F + \|\mathbf{V}\|^2_F)$$

Where:
- $C_{ij} = 1 + \alpha \cdot R_{ij}$ is the confidence weight
- $\alpha$ is the confidence scaling factor (set to 15)
- $\lambda$ is the regularization parameter (set to 0.1)
- $\|\cdot\|_F$ denotes the Frobenius norm

The confidence weighting mechanism ensures that observed interactions (visited airports) are weighted more heavily than unobserved ones, reflecting the asymmetric nature of implicit feedback.

#### ALS Update Rule

The ALS algorithm optimizes the loss function by alternately solving for user and item latent factors. For a fixed set of item factors $\mathbf{V}$, the optimal user factor $\mathbf{u}_i$ is found by solving:

$$(\mathbf{V}^T \mathbf{C}_i \mathbf{V} + \lambda \mathbf{I}) \mathbf{u}_i = \mathbf{V}^T \mathbf{C}_i \mathbf{p}_i$$

Where:
- $\mathbf{C}_i$ is a diagonal confidence matrix for user $i$
- $\mathbf{p}_i$ is the binary preference vector (1 for visited airports)
- $\mathbf{I}$ is the identity matrix

This yields a system of $k$ linear equations that can be solved efficiently using standard numerical methods.

#### Folding-In for New Profiles

For generating recommendations for a new user (a composite profile built from selected airports), we apply the folding-in technique. Given a set of liked item indices $S$, the new user latent vector $\mathbf{u}_{new}$ is computed as:

$$\mathbf{u}_{new} = (\mathbf{V}_S^T \mathbf{C} \mathbf{V}_S + \lambda \mathbf{I})^{-1} \mathbf{V}_S^T \mathbf{C}(\mathbf{1} + 1)$$

Where:
- $\mathbf{V}_S$ represents the factor matrices of the selected airports
- $\mathbf{C}$ contains confidence weights $\alpha$ for the liked items
- The recommendation score is computed as: $\text{score}_j = \mathbf{u}_{new}^T \mathbf{v}_j$

### 2. Jaccard Similarity for Tourism Preference Matching

To incorporate explicit tourism preferences, we employ the **Jaccard similarity coefficient**, a set-based similarity metric:

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}$$

Where:
- $A$ is the set of user's tourism preferences (e.g., {beach, culture, nightlife})
- $B$ is the set of tourism attributes of a candidate airport
- The resulting similarity score ranges from 0 (no overlap) to 1 (identical preferences)

### 3. Hybrid Scoring Function

The final recommendation score combines both signals through a weighted linear combination:

$$\text{Score}_{\text{final}} = \omega \cdot \text{normalize}(\text{WMF}_{\text{score}}) + (1-\omega) \cdot J(A, B)$$

Where:
- $\omega = 0.6$ weights the WMF-based collaborative signal (60%)
- $(1-\omega) = 0.4$ weights the tourism similarity signal (40%)
- The WMF score is normalized to the $[0, 1]$ range to ensure fair blending:

$$\text{normalize}(\text{WMF}_{\text{score}}) = \min\left(1.0, \frac{\text{WMF}_{\text{score}}}{\text{WMF}_{\text{score}} + \epsilon}\right)$$


This hybrid approach balances discovered patterns from user behavior with explicit preference signals.


## Project Structure

```
nextflight-route-recommender/
├── README.md                        # This file (project documentation)
├── requirements.txt                 # Python package dependencies
├── main.py                         # FastAPI backend server
├── index.html                      # Frontend UI (vanilla JS + HTML5)
├── generate_tourism_csv.py         # Script to generate tourism attributes
│
├── data/
│   ├── airports.dat                # Airport database (OpenFlights format)
│   │                               # Columns: ID, Name, City, Country, IATA, ICAO, Lat, Lon, Alt, UTC, DST
│   ├── routes.dat                  # Flight routes (optional for future features)
│   └── airport_tourism.csv         # Tourism attributes per airport
│                                   # Columns: iata, beach, mountain, culture, business, ...
│
├── model/
│   ├── wmf_model.npz               # Serialized WMF model (trained latent factors)
│   │                               # Contains: user_vectors, item_vectors
│   ├── mappings.pkl                # Index mappings (IATA ↔ latent indices)
│   │                               # Contains: airport_to_idx, idx_to_airport
│   └── recommender_route.ipynb     # Jupyter notebook for model training
│
├── info/                           # Virtual environment (created by venv)
│   ├── bin/                        # Executable scripts (python, pip, etc.)
│   ├── lib/                        # Python packages
│   └── include/                    # C headers
│
└── __pycache__/                    # Python cache files (auto-generated)
```

### File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application with recommendation endpoints |
| `index.html` | Single-page application frontend |
| `data/airports.dat` | 5,000+ airports with coordinates and names |
| `data/airport_tourism.csv` | Tourism attributes for each airport |
| `model/wmf_model.npz` | Trained WMF latent factors (~15MB) |
| `model/mappings.pkl` | Bidirectional IATA ↔ index mappings |

---

## Performance Considerations

### Memory Optimization
- Sparse matrix representation for interaction data
- Lazy loading of large datasets
- In-memory caching of precomputed factors

### Scalability
- Horizontal scaling: Deploy multiple server instances
- Database integration: Replace in-memory models with distributed backends
- Batch processing: Asynchronous recommendation computation

## Sources
- Dataset: https://github.com/jpatokal/openflights
- Slides from the course
