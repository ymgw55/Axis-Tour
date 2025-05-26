project_name="axistour"
device_id=0
container_name="${project_name}_${device_id}"
docker_image="${USER}/${project_name}"

## Set OPENAI API key
# OPENAI_API_KEY="sk-***"

docker run --rm -it --name ${container_name} \
-u $(id -u):$(id -g) \
--gpus device=${device_id} \
-v $PWD:/workspace \
-e OPENAI_API_KEY=${OPENAI_API_KEY} \
${docker_image} bash