# git clone https://github.com/magenta/ddsp
cd ddsp
apt update;
apt-get -y install ffmpeg;
pip install -U pip;
pip install tensorflow-cpu==2.5.0;
pip install -e .[data_preparation,test] --use-deprecated=legacy-resolver
cd ..
pip install flask