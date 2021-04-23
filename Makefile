DESTDIR=server

all: start
	@echo "Done"

docker-all: docker-build docker-start
	@echo "DONE"

docker-build:
	@echo "building the image from docker file..."
	docker build --pull -t project .
	@echo "image DONE"

docker-start:
	@echo "starting the NEW service in container..."
	docker run  -p 8080:8080 project

docker-vol:
	@echo "starting service in container, link to folder..."
	docker run -p 8080:8080 -v C:\Users\thayd\OneDrive\Documents\ENGR-E222\github\ML_project:/ML_project -it projectml

service:
	@echo "creating the service..."
	pip install --upgrade pip
	pip install -r requirements.txt

start:  
	@echo "starting the NEW service..."
	pip install --upgrade pip
	pip install -r requirements.txt
	python server.py

docker-stop:
	@echo "stoping the service..."
	docker stop $$(docker ps -alq)
	@echo "service stopped"

docker-remove:
	@echo "removing the image..."
	docker rmi -f project
	@echo "image removed"

docker-clean: docker-stop docker-remove
	@echo "DONE"

clean:
	@echo "removing service files created"
	rm -rf $(CREATED)
