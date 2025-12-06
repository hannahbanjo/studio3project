import shlex
import subprocess
from pathlib import Path
import os
import modal

# Paths
streamlit_script_local_path = Path(__file__).parent / "streamlit_run.py"
streamlit_script_remote_path = "/root/streamlit_run.py"

# Modal image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.9")
    .pip_install(
        "streamlit",
        "pandas",
        "matplotlib",
        "plotly",
        "supabase",
        "numpy",
        "datetime",
        "PIL"
    )
    .add_local_file(streamlit_script_local_path, streamlit_script_remote_path)
)

# Secret for Supabase credentials
secret = modal.Secret.from_name("actualmodalsecret")

app = modal.App(
    name="usage-dashboard",
    image=image,
    secrets=[secret]
)

# Make sure script exists
if not streamlit_script_local_path.exists():
    raise RuntimeError("Your streamlit_run.py file is missing.")

@app.function(allow_concurrent_inputs=100)
@modal.web_server(8000)
def run():
    """Run Streamlit inside Modal"""
    target = shlex.quote(streamlit_script_remote_path)
    cmd = (
        f"streamlit run {target} "
        f"--server.port 8000 "
        f"--server.enableCORS=false "
        f"--server.enableXsrfProtection=false"
    )
    subprocess.Popen(cmd, shell=True, env=os.environ)


# Deploy
if __name__ == "__main__":
    app.deploy()
