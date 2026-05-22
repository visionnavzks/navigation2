import sys
sys.path.insert(0, 'my/kinematic_smoother/build')
sys.path.insert(0, 'my/kinematic_smoother')
try:
    import app
    print("App imported successfully")
except Exception as e:
    print(f"Error importing app: {e}")
