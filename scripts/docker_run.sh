project_name="axistour"
device_id=0
container_name="${project_name}_${device_id}"
docker_image="${USER}/${project_name}"

## Set OPENAI API key
# OPENAI_API_KEY="sk-***"

## Set the DOCKER_HOME to specify the path of the directory to be mounted as the home directory inside the Docker container
# DOCKER_HOME=path/to/your/docker_home

docker run --rm -it --name ${container_name} \
-u $(id -u):$(id -g) \
--gpus device=${device_id} \
-v ${DOCKER_HOME}:/home/${USER} \
-v $PWD:/workspace \
-e OPENAI_API_KEY=${OPENAI_API_KEY} \
${docker_image} bash