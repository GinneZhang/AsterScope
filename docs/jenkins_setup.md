# Jenkins Setup

This repository supports a production-minded Jenkins flow with:

- pull request validation only
- `main` branch image build and push
- Helm-based Kubernetes deployment
- separate `staging` and `prod` deployment modes

## Recommended Jenkins Jobs

### 1. Multibranch Pipeline

Create a Jenkins **Multibranch Pipeline** pointing at:

- `https://github.com/GinneZhang/AsterScope.git`

Recommended branch behavior:

- pull requests:
  - run `Sanity`
  - run `Unit Tests`
  - optionally run `Integration Tests`
  - do not build or deploy
- `main`:
  - run tests
  - build and push `api` and `retrieval` images
  - deploy through Helm

### 2. GitHub Webhook

In the GitHub repository webhook settings, point the webhook to:

- `https://<your-jenkins-host>/github-webhook/`

Recommended events:

- `Just the push event`
- `Pull requests`

### 3. Build Status Badge

Replace `<your-jenkins-host>` with your real Jenkins host and use one of these badge URLs.

Classic pipeline badge:

```text
https://<your-jenkins-host>/buildStatus/icon?job=AsterScope%2Fmain
```

Markdown example:

```md
![Jenkins](https://<your-jenkins-host>/buildStatus/icon?job=AsterScope%2Fmain)
```

If you use a multibranch pipeline, the exact job path may differ. Copy the badge URL directly from the Jenkins job page once the pipeline exists.

## Required Jenkins Credentials

- `docker-registry-creds`
  - username/password credential for your container registry
- `kubeconfig-asterscope`
  - secret file credential containing kubeconfig for the target cluster

## Recommended Parameters

- `RUN_INTEGRATION_TESTS=true`
- `DEPLOY_TO_K8S=true`
- `DEPLOY_ENVIRONMENT=staging` or `prod`
- `K8S_NAMESPACE=`
- `HELM_RELEASE_NAME=`
- `API_IMAGE_REPOSITORY=asterscope/api`
- `RETRIEVAL_IMAGE_REPOSITORY=asterscope/retrieval`
- `DOCKER_REGISTRY=ghcr.io/<your-org>` or your private registry

## Helm Deployment Behavior

The Jenkins deploy step uses:

- [scripts/jenkins/deploy_stack.sh](/Users/ginnezhang/Documents/Playground/NovaSearch/scripts/jenkins/deploy_stack.sh)
- [deploy/helm/asterscope](/Users/ginnezhang/Documents/Playground/NovaSearch/deploy/helm/asterscope)

Environment overlays:

- [deploy/helm/asterscope/values-staging.yaml](/Users/ginnezhang/Documents/Playground/NovaSearch/deploy/helm/asterscope/values-staging.yaml)
- [deploy/helm/asterscope/values-prod.yaml](/Users/ginnezhang/Documents/Playground/NovaSearch/deploy/helm/asterscope/values-prod.yaml)

Production deployments require manual confirmation in Jenkins before rollout.
