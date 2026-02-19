
# CONFIG
PROJECT_NAME := multimodalai-mlops
ENV_FILE := .env

# FILE PATHS
FILE_STORAGE := ./docker/01_storage/docker-compose.yaml
FILE_TRACKING := ./docker/02_tracking/docker-compose.yaml
FILE_ORCHESTRATION := ./docker/03_orchestration/docker-compose.yaml
FILE_INFERENCE := ./docker/04_inference/docker-compose.yaml

# Docker Compose command
COMPOSE_CMD := docker compose \
	--env-file $(ENV_FILE) \
	-p $(PROJECT_NAME) \
	-f $(FILE_STORAGE) \
	-f $(FILE_TRACKING) \
	-f $(FILE_ORCHESTRATION) \
	-f $(FILE_INFERENCE)

	# Par défaut, si on tape juste "make", on affiche l'aide
.DEFAULT_GOAL := help

help:  ## Show this help message
	@echo "Available commands for $(PROJECT_NAME):"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

up:  ## Start the stack (force build)
		@echo "Starting $(PROJECT_NAME)..."
		$(COMPOSE_CMD) up -d
		@echo "✅ Stack is up! Services available at:"
		@echo "   - Airflow   : http://localhost:8080"
		@echo "   - MLflow    : http://localhost:5000"
		@echo "   - MinIO     : http://localhost:9001"
		@echo "   - Inference : http://localhost:8000"

rebuild:  ## Rebuild the stack
		@echo "Rebuilding $(PROJECT_NAME)..."
		$(COMPOSE_CMD) build --no-cache

down:  ## Stop the stack and remove containers
		@echo "🛑 Stopping $(PROJECT_NAME)..."
		$(COMPOSE_CMD) down
		@echo "Stack Stopped."

logs:  ## Show live logs (Ctrl+C to exit)
		$(COMPOSE_CMD) logs -f

status:  ## Show stack status
		$(COMPOSE_CMD) ps

config:  ## Validate Docker Compose configuration
		$(COMPOSE_CMD) config
