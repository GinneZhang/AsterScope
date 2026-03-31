pipeline {
  agent any

  options {
    timestamps()
    ansiColor('xterm')
    disableConcurrentBuilds()
    buildDiscarder(logRotator(numToKeepStr: '20'))
  }

  parameters {
    booleanParam(name: 'RUN_INTEGRATION_TESTS', defaultValue: true, description: 'Start docker-compose infra and run integration tests.')
    booleanParam(name: 'DEPLOY_TO_K8S', defaultValue: true, description: 'Deploy the API and retrieval images to Kubernetes after a successful main-branch build.')
    string(name: 'K8S_NAMESPACE', defaultValue: 'asterscope', description: 'Kubernetes namespace for deployment.')
    string(name: 'API_IMAGE_REPOSITORY', defaultValue: 'asterscope/api', description: 'Container image repository for the API service.')
    string(name: 'RETRIEVAL_IMAGE_REPOSITORY', defaultValue: 'asterscope/retrieval', description: 'Container image repository for the retrieval service.')
    string(name: 'IMAGE_TAG', defaultValue: '', description: 'Optional image tag override. Defaults to Jenkins BUILD_NUMBER.')
    string(name: 'DOCKER_REGISTRY', defaultValue: '', description: 'Optional registry prefix, e.g. ghcr.io/your-org')
    string(name: 'DOCKER_REGISTRY_CREDENTIALS_ID', defaultValue: 'docker-registry-creds', description: 'Jenkins username/password credential id for the container registry.')
    string(name: 'KUBECONFIG_CREDENTIALS_ID', defaultValue: 'kubeconfig-asterscope', description: 'Jenkins secret file credential id containing kubeconfig.')
    string(name: 'HELM_RELEASE_NAME', defaultValue: 'asterscope', description: 'Helm release name for the deployed stack.')
  }

  environment {
    PYTHONUNBUFFERED = '1'
    PIP_DISABLE_PIP_VERSION_CHECK = '1'
    DOCKER_BUILDKIT = '1'
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Setup Python') {
      steps {
        sh '''
          python3 -m venv .venv
          . .venv/bin/activate
          python -m pip install --upgrade pip
          pip install -r requirements.txt -r requirements_dev.txt
        '''
      }
    }

    stage('Sanity') {
      steps {
        sh '''
          . .venv/bin/activate
          python -m py_compile api/main.py agent/planner.py core/memory.py retrieval/graph/cypher_generator.py
          bash -n scripts/jenkins/run_ci_tests.sh
          bash -n scripts/jenkins/wait_for_stack.sh
          bash -n scripts/jenkins/deploy_api.sh
          bash -n scripts/jenkins/deploy_retrieval.sh
          bash -n scripts/jenkins/deploy_stack.sh
        '''
      }
    }

    stage('Unit Tests') {
      steps {
        sh '''
          . .venv/bin/activate
          ./scripts/jenkins/run_ci_tests.sh unit
        '''
      }
      post {
        always {
          junit allowEmptyResults: true, testResults: 'reports/unit.xml'
        }
      }
    }

    stage('Integration Tests') {
      when {
        expression { return params.RUN_INTEGRATION_TESTS }
      }
      steps {
        sh '''
          docker compose up -d postgres redis neo4j elasticsearch
          ./scripts/jenkins/wait_for_stack.sh
          . .venv/bin/activate
          ./scripts/jenkins/run_ci_tests.sh integration
        '''
      }
      post {
        always {
          junit allowEmptyResults: true, testResults: 'reports/integration.xml'
          sh 'docker compose down -v || true'
        }
      }
    }

    stage('Build Images') {
      when {
        branch 'main'
      }
      steps {
        script {
          env.RESOLVED_IMAGE_TAG = params.IMAGE_TAG?.trim() ? params.IMAGE_TAG.trim() : env.BUILD_NUMBER
          env.API_IMAGE_NAME = params.DOCKER_REGISTRY?.trim()
            ? "${params.DOCKER_REGISTRY.trim()}/${params.API_IMAGE_REPOSITORY}:${env.RESOLVED_IMAGE_TAG}"
            : "${params.API_IMAGE_REPOSITORY}:${env.RESOLVED_IMAGE_TAG}"
          env.RETRIEVAL_IMAGE_NAME = params.DOCKER_REGISTRY?.trim()
            ? "${params.DOCKER_REGISTRY.trim()}/${params.RETRIEVAL_IMAGE_REPOSITORY}:${env.RESOLVED_IMAGE_TAG}"
            : "${params.RETRIEVAL_IMAGE_REPOSITORY}:${env.RESOLVED_IMAGE_TAG}"
        }
        sh '''
          echo "${API_IMAGE_NAME}" > .image_api_ref
          echo "${RETRIEVAL_IMAGE_NAME}" > .image_retrieval_ref
          docker build -t "${API_IMAGE_NAME}" -t "${RETRIEVAL_IMAGE_NAME}" .
        '''
      }
    }

    stage('Push Images') {
      when {
        branch 'main'
      }
      steps {
        withCredentials([usernamePassword(credentialsId: params.DOCKER_REGISTRY_CREDENTIALS_ID, usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
          sh '''
            if [ -n "${DOCKER_REGISTRY}" ]; then
              echo "${DOCKER_PASS}" | docker login "${DOCKER_REGISTRY}" -u "${DOCKER_USER}" --password-stdin
            else
              echo "${DOCKER_PASS}" | docker login -u "${DOCKER_USER}" --password-stdin
            fi
            docker push "$(cat .image_api_ref)"
            docker push "$(cat .image_retrieval_ref)"
          '''
        }
      }
    }

    stage('Deploy Services') {
      when {
        allOf {
          branch 'main'
          expression { return params.DEPLOY_TO_K8S }
        }
      }
      steps {
        withCredentials([file(credentialsId: params.KUBECONFIG_CREDENTIALS_ID, variable: 'KUBECONFIG')]) {
          sh '''
            chmod +x scripts/jenkins/deploy_stack.sh
            ./scripts/jenkins/deploy_stack.sh "$(cat .image_api_ref)" "$(cat .image_retrieval_ref)" "${K8S_NAMESPACE}" "${HELM_RELEASE_NAME}"
          '''
        }
      }
    }
  }
}
