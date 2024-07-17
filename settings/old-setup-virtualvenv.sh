
echo "Installing Python and env"

sudo apt update --fix-missing -y 

sudo apt install -y --no-install-recommends apt-utils module-init-tools software-properties-common build-essential

sudo add-apt-repository -y ppa:deadsnakes/ppa

sudo apt update 

sudo apt install -y python3.6 python3.6-dev python3-pip python3-venv 

#ln -snf /usr/share/zoneinfo/$TZ /etc/localtime 
#echo $TZ > /etc/timezone
#
#sudo apt install -y --no-install-recommends module-init-tools wget nano curl git ninja-build ccache



echo "Setting ISENV"
python3.6 -m venv env

printf "\nexport ATCISELWORKDIR=`dirname $PWD`" >> env/bin/activate

source env/bin/activate

echo "ATCISELWORKDIR = ${ATCISELWORKDIR}"

pip install --upgrade pip wheel setuptools

pip install -r requirements.txt

deactivate


