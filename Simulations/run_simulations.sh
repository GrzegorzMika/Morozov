docker build -t beta -f Dockerfile_beta .
docker build -t beta_tikhonov -f Dockerfile_beta_tikhonov .
docker build -t beta_tikhonov_1 -f Dockerfile_beta_tikhonov_1 .
docker build -t beta_tsvd -f Dockerfile_beta_tsvd .
docker build -t bimodal -f Dockerfile_bimodal .
docker build -t bimodal_tikhonov -f Dockerfile_bimodal_tikhonov .
docker build -t bimodal_tikhonov_1 -f Dockerfile_bimodal_tikhonov_1 .
docker build -t bimodal_tsvd -f Dockerfile_bimodal_tsvd .
docker build -t smla -f Dockerfile_smla .
docker build -t smla_tikhonov -f Dockerfile_smla_tikhonov .
docker build -t smla_tikhonov_1 -f Dockerfile_smla_tikhonov_1 .
docker build -t smla_tsvd -f Dockerfile_smla_tsvd .
docker build -t smlb -f Dockerfile_smlb .
docker build -t smlb_tikhonov -f Dockerfile_smlb_tikhonov .
docker build -t smlb_tikhonov_1 -f Dockerfile_smlb_tikhonov_1 .
docker build -t smlb_tsvd -f Dockerfile_smlb_tsvd .

docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined beta:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined beta_tikhonov:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined beta_tikhonov_1:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined beta_tsvd:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined bimodal:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined bimodal_tikhonov:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined bimodal_tikhonov_1:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined bimodal_tsvd:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smla:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smla_tikhonov:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smla_tikhonov_1:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smla_tsvd:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smlb:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smlb_tikhonov:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smlb_tikhonov_1:latest
docker run -d -v cachedir:/home/Morozov/cachedir --security-opt seccomp=unconfined smlb_tsvd:latest