echo "Importing opencv library from host into venv..."
# Find cv2 library for the global Python installation.
GLOBAL_CV2=$(/usr/bin/python3 -c 'import cv2; print(cv2.__file__)')
# Find site-packages directory in the venv
VENV_SITEPACKAGES_DIR=$(python3 -c 'import site; print(site.getsitepackages()[0])')
# Copy host-installed library file into venv
echo $GLOBAL_CV2
echo $VENV_SITEPACKAGES_DIR
#sudo cp ${GLOBAL_CV2} ${VENV_SITEPACKAGES_DIR}
