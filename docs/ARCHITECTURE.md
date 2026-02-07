# Architecture et Justification des Choix Technologiques

## Vue d'ensemble

Ce projet implémente un pipeline MLOps complet pour la détection d'objets, suivant les meilleures pratiques de l'industrie pour la reproductibilité, la traçabilité et le déploiement continu.

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE GLOBALE                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│    ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐  │
│    │    DATA      │     │   TRAINING   │     │   TRACKING   │     │   SERVING    │  │
│    │              │     │              │     │              │     │              │  │
│    │  COCO128     │────▶│   YOLO11     │────▶│   MLflow     │────▶│   FastAPI    │  │
│    │  + DVC       │     │   Optuna     │     │   Registry   │     │   /predict   │  │
│    └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘  │
│           │                    │                    │                    │           │
│           ▼                    ▼                    ▼                    ▼           │
│    ┌──────────────────────────────────────────────────────────────────────────────┐ │
│    │                           INFRASTRUCTURE                                      │ │
│    │   MinIO (S3)  │  PostgreSQL  │  Docker Compose  │  GitHub Actions            │ │
│    └──────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Modèle de Détection : YOLO11 (Ultralytics)

### Choix
**YOLO11n** (nano) - le modèle le plus léger de la famille YOLO11.

### Justification

| Critère | YOLO11 | Detectron2 | DETR | MMDetection |
|---------|--------|------------|------|-------------|
| **Facilité d'intégration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Performance CPU** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Documentation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **MLflow natif** | ✅ Intégré | ❌ Manuel | ❌ Manuel | ❌ Manuel |
| **Export ONNX/TensorRT** | ✅ Simple | ⚠️ Complexe | ⚠️ Complexe | ⚠️ Complexe |

### Avantages
1. **API unifiée** : `model.train()`, `model.val()`, `model.predict()` - cohérent et simple
2. **Logging MLflow natif** : Intégration automatique sans code supplémentaire
3. **Poids pré-entraînés** : Téléchargement automatique depuis les releases GitHub
4. **Performance** : 6.6 GFLOPs pour yolo11n, idéal pour CPU
5. **Écosystème mature** : Documentation extensive, communauté active

### Alternatives considérées
- **Detectron2** : Plus puissant mais complexité accrue, moins adapté au CPU
- **DETR** : Architecture Transformer, excellent mais lent en inférence
- **YOLOv5/v8** : Versions précédentes, YOLO11 apporte des améliorations

---

## 2. Dataset : COCO128

### Choix
COCO128 - sous-ensemble de 128 images du dataset COCO.

### Justification

| Critère | COCO128 | COCO Full | Pascal VOC | Custom |
|---------|---------|-----------|------------|--------|
| **Taille** | 6.7 MB | 25 GB | 2 GB | Variable |
| **Classes** | 80 | 80 | 20 | Variable |
| **Téléchargement** | < 10s | > 1h | ~10min | N/A |
| **Démonstration** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Avantages
1. **Rapidité** : Permet de démontrer le pipeline complet en minutes
2. **Représentativité** : Mêmes 80 classes que COCO complet
3. **Intégration Ultralytics** : Téléchargement automatique via `yolo.check('coco128')`
4. **Reproductibilité** : Dataset public et stable

### Pour production
Remplacer par COCO complet ou dataset custom :
```bash
docker compose run trainer python -m src.data.ingest --dataset coco --data-dir /app/data/coco
```

---

## 3. Versionnement des Données : DVC

### Choix
DVC (Data Version Control) avec remote S3 (MinIO).

### Justification

| Critère | DVC | Git LFS | Delta Lake | Pachyderm |
|---------|-----|---------|------------|-----------|
| **Intégration Git** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Coût** | Gratuit | Limité | Gratuit | Payant |
| **Courbe d'apprentissage** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Remote S3** | ✅ Natif | ❌ | ✅ | ✅ |
| **Pipeline ML** | ✅ `dvc.yaml` | ❌ | ❌ | ✅ |

### Architecture DVC
```
Git Repository                    MinIO (S3)
├── data/                         s3://dvc/
│   ├── coco128.dvc  ───────────▶ ├── files/md5/ab/cdef12345...
│   └── .gitignore                └── files/md5/78/90abcdef...
├── .dvc/
│   └── config       (remote URL)
```

### Avantages
1. **Séparation Git/Data** : Les données binaires ne polluent pas Git
2. **Reproductibilité** : `git checkout v1 && dvc checkout` restaure l'exact état
3. **Remote flexible** : S3, GCS, Azure, SSH, local
4. **Pipeline DAG** : Possibilité de définir des dépendances (non utilisé ici car ZenML)

---

## 4. Tracking & Registry : MLflow

### Choix
MLflow avec backend PostgreSQL et artifacts sur MinIO (S3).

### Justification

| Critère | MLflow | Weights & Biases | Neptune | Comet |
|---------|--------|------------------|---------|-------|
| **Self-hosted** | ✅ Gratuit | ❌ Cloud | ❌ Cloud | ❌ Cloud |
| **Model Registry** | ✅ Intégré | ✅ | ✅ | ✅ |
| **UI** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **API REST** | ✅ | ✅ | ✅ | ✅ |
| **Intégration YOLO** | ✅ Natif | ⚠️ Plugin | ⚠️ Plugin | ⚠️ Plugin |

### Architecture MLflow
```
┌─────────────────────────────────────────────────────────────┐
│                     MLflow Server                           │
│                    (http://localhost:5000)                  │
├────────────────────────┬────────────────────────────────────┤
│   Tracking Store       │        Artifact Store              │
│   (PostgreSQL)         │        (MinIO S3)                  │
│                        │                                    │
│   - Experiments        │   s3://mlflow/                     │
│   - Runs               │   ├── 0/{run_id}/                  │
│   - Parameters         │   │   ├── weights/best.pt          │
│   - Metrics            │   │   ├── confusion_matrix.png     │
│   - Tags               │   │   └── results.csv              │
│                        │                                    │
│   Model Registry       │                                    │
│   - detector v1        │                                    │
│   - detector v2        │                                    │
└────────────────────────┴────────────────────────────────────┘
```

### Ce qui est loggé
```python
# Paramètres
mlflow.log_params({
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 0.01,
    "model": "yolo11n.pt",
    "imgsz": 640
})

# Métriques
mlflow.log_metrics({
    "mAP50": 0.683,
    "mAP50-95": 0.515,
    "precision": 0.72,
    "recall": 0.68
})

# Artifacts
mlflow.log_artifact("weights/best.pt")
mlflow.log_artifact("confusion_matrix.png")

# Reproducibilité
mlflow.set_tags({
    "git_commit": "abc123",
    "dvc_data_version": "xyz789",
    "config_hash": "def456"
})
```

---

## 5. Optimisation Hyperparamètres : Optuna

### Choix
Optuna avec pruning MedianPruner.

### Justification

| Critère | Optuna | Ray Tune | Hyperopt | Keras Tuner |
|---------|--------|----------|----------|-------------|
| **Algorithme** | TPE, CMA-ES | ASHA, PBT | TPE | Bayesian |
| **Pruning** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Visualisation** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **Légèreté** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

### Espace de recherche
```yaml
# configs/optuna.yaml
search_space:
  learning_rate:
    type: float
    low: 0.001
    high: 0.1
    log: true
  
  batch_size:
    type: categorical
    choices: [8, 16, 32]
  
  optimizer:
    type: categorical
    choices: ["SGD", "Adam", "AdamW"]
```

### Avantages
1. **Pruning automatique** : Arrête les trials non prometteurs tôt
2. **Define-by-run** : API Pythonique fluide
3. **Persistance SQLite/PostgreSQL** : Reprise après interruption
4. **Intégration MLflow** : Chaque trial est un run MLflow

---

## 6. Orchestration : ZenML

### Choix
ZenML pipeline local.

### Justification

| Critère | ZenML | Airflow | Kubeflow | Prefect |
|---------|-------|---------|----------|---------|
| **ML-first** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Local dev** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **Artifacts** | ✅ Natif | ❌ | ✅ | ⚠️ |
| **Démarrage** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐ |

### Pipeline défini
```python
@pipeline
def training_pipeline(config_path: str):
    """
    Pipeline: Ingest → Train → Eval → Register
    """
    data_path = ingest_data()
    model_path, metrics = train_model(data_path, config_path)
    eval_metrics = evaluate_model(model_path, data_path)
    register_model(model_path, metrics)
```

### Avantages
1. **ML-native** : Conçu pour le ML, pas le data engineering
2. **Artifact tracking** : Chaque step produit des artifacts tracés
3. **Local-first** : Fonctionne sans infrastructure externe
4. **Extensible** : Peut déployer sur Kubernetes, Sagemaker, etc.

---

## 7. Serving : FastAPI

### Choix
FastAPI avec chargement dynamique de modèle depuis MLflow.

### Justification

| Critère | FastAPI | Flask | Django | TorchServe |
|---------|---------|-------|--------|------------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Async** | ✅ Natif | ⚠️ | ⚠️ | ✅ |
| **OpenAPI** | ✅ Auto | ❌ | ❌ | ✅ |
| **Validation** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

### Endpoints
```
GET  /health       → HealthResponse
POST /predict      → DetectionResponse (multipart/form-data)
GET  /model/info   → ModelInfo
GET  /metrics      → Prometheus metrics
```

### Rollback instantané
```bash
# Déployer v1
MODEL_URI=models:/detector/1 docker compose up -d api

# Rollback en 3 secondes
docker compose down api
MODEL_URI=models:/detector/1 docker compose up -d api
```

---

## 8. Infrastructure : Docker Compose

### Choix
Docker Compose pour l'orchestration des services.

### Services

| Service | Port | Rôle |
|---------|------|------|
| **minio** | 9000, 9001 | Stockage S3 (artifacts, data) |
| **postgres** | 5432 | Backend MLflow |
| **mlflow** | 5000 | Tracking + Model Registry |
| **trainer** | - | Container d'entraînement |
| **api** | 8000 | Serving API |
| **prometheus** | 9090 | Métriques (optionnel) |
| **grafana** | 3000 | Dashboards (optionnel) |

### Justification
1. **Isolation** : Chaque service dans son container
2. **Reproductibilité** : Même environnement dev/prod
3. **Zero install** : Seul Docker requis sur l'hôte
4. **Networking** : Réseau privé `mlops-net`

---

## 9. CI/CD : GitHub Actions

### Pipeline
```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│  Push   │───▶│   Lint   │───▶│  Test   │───▶│  Build   │
└─────────┘    └──────────┘    └─────────┘    └──────────┘
                                                    │
                                                    ▼
                                             ┌──────────┐
                                             │  Deploy  │ (on tag v*)
                                             └──────────┘
```

### Jobs
1. **lint** : Ruff linter + formatter check
2. **test** : pytest sur tests/test_api.py
3. **smoke-test** : Test d'entraînement 1 epoch (weekly)
4. **build** : Build + push Docker images vers ghcr.io
5. **deploy** : Déploiement sur tag v*

---

## 10. Structure de Reproductibilité

### Ce qui est versionné

| Artefact | Outil | Stockage |
|----------|-------|----------|
| Code source | Git | GitHub |
| Données | DVC | MinIO (s3://dvc) |
| Modèles | MLflow | MinIO (s3://mlflow) |
| Configuration | Git | GitHub |
| Métriques | MLflow | PostgreSQL |
| Conteneurs | Docker | ghcr.io |

### Reproduction d'une expérience
```bash
# 1. Checkout exact code version
git checkout <git_commit>

# 2. Restore exact data version
dvc checkout

# 3. Re-run with same config
docker compose run trainer python -m src.train.train \
    --config /app/configs/train_baseline.yaml
```

### Tags MLflow pour traçabilité
```
git_commit: abc123def456
dvc_data_version: xyz789abc012
config_hash: 99255cb3
python_version: 3.11.14
ultralytics_version: 8.4.12
```

---

## Conclusion

Cette architecture suit les principes **ML Engineering** modernes :

1. ✅ **Reproductibilité** : Git + DVC + MLflow tags
2. ✅ **Traçabilité** : Chaque run lié à code/data/config
3. ✅ **Automatisation** : CI/CD GitHub Actions
4. ✅ **Isolation** : Docker Compose, zero install host
5. ✅ **Évolutivité** : Peut migrer vers Kubernetes
6. ✅ **Rollback** : Changement de modèle instantané

### Prochaines améliorations possibles
- [ ] Kubernetes (Helm chart)
- [ ] A/B testing avec Istio
- [ ] Feature store (Feast)
- [ ] Monitoring ML (Evidently, Seldon)
- [ ] Auto-retraining sur drift
