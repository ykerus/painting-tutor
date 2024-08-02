
include .env

IMAGE_NAME := ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT_ID}/docker/painting-tutor

.PHONY: run-local
run-local:
	streamlit run src/painting_tutor/app/app.py

.PHONY: docker-build
docker-build:
	docker build --platform linux/amd64 -t ${IMAGE_NAME} .

.PHONY: docker-push
docker-push:
	docker push ${IMAGE_NAME}

.PHONY: docker-run
docker-run:
	docker run --env-file .env --rm -p 8080:8080 ${IMAGE_NAME}

.PHONY: cloud-run-deploy
cloud-run-deploy:
	gcloud run deploy painting-tutor \
		--image ${IMAGE_NAME} \
		--region ${GCP_REGION} \
		--port 8080 \
		--min-instances 0 \
		--max-instances 1 \
		--memory 1Gi \
		--set-env-vars APP_PASSWORD=${APP_PASSWORD} \
		--allow-unauthenticated

.PHONY: build-push-deploy
build-push-deploy: docker-build docker-push cloud-run-deploy
