# Next Trip Route Recommender

A flight route recommendation system leveraging **Weighted Matrix Factorization (WMF)** with collaborative filtering and semantic similarity matching. This project combines latent factor models with tourism preference analysis to deliver personalized airport recommendations based on user travel history.

**Live Demo**: Open `index.html` in your browser after starting the backend server.

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
- $k$ is the number of latent factors (set to 100 in this implementation)

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

## Troubleshooting & Common Issues

### ❌ Issue: "ModuleNotFoundError: No module named 'fastapi'"

**Cause**: Virtual environment not activated or dependencies not installed

**Solution**:
```bash
# Verify virtual environment is activated (should see (info) in terminal)
source info/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python3 -c "import fastapi; print('✓ fastapi installed')"
```

---

### ❌ Issue: "Address already in use" on port 8000

**Cause**: Another process is using port 8000

**Solution Option 1 - Kill the existing process**:
```bash
# macOS/Linux
lsof -ti:8000 | xargs kill -9

# Windows (PowerShell)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process
```

**Solution Option 2 - Use a different port**:
```bash
uvicorn main:app --host 127.0.0.1 --port 8001 --reload
# Then access at http://127.0.0.1:8001
```

---

### ❌ Issue: "FileNotFoundError: airports.dat not found"

**Cause**: Data files are missing from the `data/` directory

**Solution**:
```bash
# Verify file exists
ls -la data/airports.dat

# Check file permissions
chmod 644 data/airports.dat

# Verify file encoding (should be UTF-8)
file data/airports.dat

# Verify file is not empty
wc -l data/airports.dat
```

---

### ❌ Issue: "Frontend shows 'Error: Unable to connect to the FastAPI server'"

**Cause**: Backend server is not running or port is incorrect

**Solution**:
```bash
# 1. Verify backend is running
curl http://127.0.0.1:8000/airports

# 2. Check if API responds
curl http://127.0.0.1:8000/docs

# 3. If connection refused, start backend
source info/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# 4. If using a different port, update index.html
# Change all "http://127.0.0.1:8000" to "http://127.0.0.1:8001"
```

---

### ❌ Issue: "CORS error in browser console"

**Cause**: Frontend and backend on different ports without proper CORS headers

**Solution**: Already configured in `main.py`, but if issues persist:
```python
# In main.py, verify this exists:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

### ❌ Issue: "recommendations/airports endpoint returns empty list"

**Cause**: Model files not loaded properly

**Solution**:
```bash
# Verify model files exist
ls -la model/wmf_model.npz
ls -la model/mappings.pkl

# Check file sizes (should be >1MB)
ls -lh model/

# Restart backend (in case of partial load)
pkill -f uvicorn
source info/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

### ✅ Debugging Checklist

Use this checklist to diagnose issues:

```bash
# 1. Virtual environment activated?
which python3  # Should show path inside 'info' directory

# 2. Dependencies installed?
pip list | grep -E "fastapi|uvicorn|numpy"

# 3. Backend running?
curl -s http://127.0.0.1:8000/docs | grep -i "swagger"

# 4. Data files present?
[ -f data/airports.dat ] && echo "✓ airports.dat" || echo "✗ Missing"
[ -f data/airport_tourism.csv ] && echo "✓ airport_tourism.csv" || echo "✗ Missing"

# 5. Model files present?
[ -f model/wmf_model.npz ] && echo "✓ wmf_model.npz" || echo "✗ Missing"
[ -f model/mappings.pkl ] && echo "✓ mappings.pkl" || echo "✗ Missing"

# 6. Port conflict?
lsof -i :8000 | grep -i listen && echo "✗ Port 8000 in use" || echo "✓ Port available"
```

---

## 🔧 Development Workflow

### Making Changes to the Backend

1. **Edit `main.py`** with your changes
2. **Backend reloads automatically** (due to `--reload` flag)
3. **Test with curl or the frontend**:

```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"airports": ["MAD", "BCN"]}'
```

### Making Changes to the Frontend

1. **Edit `index.html`** with your changes
2. **Refresh your browser** (Ctrl+R or Cmd+R)
3. **Check browser console** (F12 → Console) for errors

### Adding New Endpoints

Example: Add a new endpoint `/health-check`

```python
@app.get("/health-check")
async def health_check():
    """Simple endpoint to verify server is running"""
    return {"status": "healthy", "timestamp": str(datetime.now())}
```

Test it:
```bash
curl http://127.0.0.1:8000/health-check
```

---

## References

- **WMF with ALS**: Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative filtering for implicit feedback datasets*. ICDM.
- **Folding-In Technique**: Koren, Y. (2010). *Collaborative filtering with temporal dynamics*. KDD.
- **Jaccard Similarity**: Jaccard, P. (1912). *The distribution of flora in the alpine zone*. New Phytologist.

## License

This project is available under the MIT License. See LICENSE file for details.

```
MIT License

Copyright (c) 2026 Fernando A. Martínez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Author

**Fernando A. Martínez**
- 📧 Email: fernandeza@example.com
- 🔗 GitHub: [@afm719](https://github.com/afm719)
- 💼 LinkedIn: [Fernando Martínez](https://linkedin.com/in/fernandoa-martinez)

## Contributing

Contributions are welcome! Please follow these steps:

### 1. Fork the Repository

```bash
# Go to https://github.com/afm719/nextflight-route-recommender
# Click "Fork" button
```

### 2. Clone Your Fork

```bash
git clone https://github.com/YOUR_USERNAME/nextflight-route-recommender.git
cd nextflight-route-recommender
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-amazing-feature
```

### 4. Make Changes and Test

```bash
# Make your changes
# Test thoroughly
python3 test_api.py  # If available
```

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add amazing feature: description of what you added"
# Use clear commit messages!
```

### 6. Push to Your Fork

```bash
git push origin feature/your-amazing-feature
```

### 7. Open a Pull Request

- Go to https://github.com/afm719/nextflight-route-recommender
- Click "Compare & pull request"
- Describe your changes clearly
- Wait for review and feedback

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and modular
- Test your changes before submitting

---

## Changelog

### Version 1.0.0 (April 22, 2026)
- ✅ Initial release
- ✅ WMF-based recommendation system
- ✅ Hybrid scoring with Jaccard similarity
- ✅ Interactive frontend with real-time updates
- ✅ Comprehensive API documentation
- ✅ Tourism preference filtering

### Planned Features (v1.1+)
- 🔄 Temporal dynamics (seasonal patterns)
- 🗺️ Geographic clustering
- 💰 Price integration
- 👥 Social filtering
- 🧠 Deep learning models
- 📊 A/B testing framework
- 📱 Mobile app
- 🌐 Multi-language support

---

## Citation

If you use this project in academic work or research, please cite:

```bibtex
@software{martinez2026nextflight,
  author = {Martínez, Fernando A.},
  title = {Next Trip Route Recommender: A Hybrid Recommendation System for Flight Destinations},
  year = {2026},
  url = {https://github.com/afm719/nextflight-route-recommender},
  note = {Accessed: YYYY-MM-DD}
}
```

---

## Acknowledgments

- Yehuda Koren & Robert Bell for pioneering WMF research
- OpenFlights for the airport database
- FastAPI community for the excellent web framework
- All contributors and users who provide feedback

---

## Support & Contact

### Getting Help

1. **Check the troubleshooting section** above for common issues
2. **Search existing GitHub issues**: https://github.com/afm719/nextflight-route-recommender/issues
3. **Create a new issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Your Python version (`python3 --version`)
   - Error messages (full traceback)
   - Your operating system

### Feature Requests

Have an idea for improvement? 
- Open an issue with the "enhancement" label
- Describe the use case and expected behavior
- Provide examples if possible

### Contact

For inquiries not related to technical support:
- 📧 Email: fernandeza@example.com
- 💬 GitHub Discussions: (when available)

---

## Quick Reference

### Common Commands

```bash
# Activate virtual environment
source info/bin/activate

# Start backend server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Run frontend
python3 -m http.server 8080

# Test API
curl -X GET "http://127.0.0.1:8000/airports" | python3 -m json.tool

# View API docs
open http://127.0.0.1:8000/docs

# Check Python version
python3 --version

# List installed packages
pip list

# Update requirements
pip freeze > requirements.txt
```

---

**Last Updated**: April 22, 2026  
**Version**: 1.0.0  
**Status**: Active Development

🚀 **Ready to build your next feature? Let's go!**
