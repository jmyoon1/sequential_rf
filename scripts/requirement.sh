sudo add-apt-repository universe
sudo apt update
sudo apt-get -y install python-is-python3
pip install --upgrade "jax[tpu]==0.4.4" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install --upgrade flax==0.6.4
pip install --upgrade optax==0.1.4
pip install --upgrade orbax==0.1.2
pip install tensorflow
pip install tensorflow_datasets
pip install tensorflow_gan
pip install tensorflow==2.8.0
pip install tensorflow_io==0.25.0
pip install tensorflow_probability==0.16.0
pip install einops
pip install wandb==0.15.0
pip install ml_collections
pip install lpips-j
pip install transformers
pip install diffusers
pip install protobuf==3.20.3
sudo apt-get -y install golang
pip install jax-smi