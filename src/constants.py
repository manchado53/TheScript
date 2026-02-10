import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path_model = os.path.join(_REPO_ROOT, 'runs', 'pose', 'train18', 'weights', 'best.pt')
