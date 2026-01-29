# Dial Helm Chart

This Helm chart deploys DIAL (Distributed INTERSECT Active Learning) service along with MongoDB for persistent storage.

The chart is based on the [Bitnami Charts Template](https://github.com/bitnami/charts) and uses the Bitnami Common library for standardization.

## Components

- **Dial Service**: Bayesian optimization and active learning service
- **MongoDB**: Document database for storing models and workflow data (via Bitnami subchart)

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+

## Installation

### Add Bitnami Repository (if not already added)

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update
```

### Install the Chart

1. Update dependencies:
```bash
cd chart
helm dependency update
```

2. Install the chart with default values:
```bash
helm install dial . -n dial --create-namespace
```

3. Or install with custom values:
```bash
helm install dial . -n dial --create-namespace -f values.yaml -f values.config.yaml
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dial.image.repository` | Dial image repository | `intersect-fabric/dial` |
| `dial.image.tag` | Dial image tag | `latest` |
| `replicaCount` | Number of replicas | `1` |
| `mongodb.enabled` | Enable MongoDB subchart | `true` |
| `mongodb.auth.username` | MongoDB username | `dial` |
| `mongodb.auth.password` | MongoDB password | `changeme` |
| `mongodb.auth.database` | MongoDB database name | `dial` |
| `mongodb.persistence.size` | MongoDB storage size | `8Gi` |

### INTERSECT Configuration

NOTE: Not sure if this is used by DIAL, looks like JSON is way.

Configure INTERSECT-specific settings in `values.yaml`:

```yaml
intersectConfig:
  brokers:
    - host: broker-hostname
      username: BROKER_USER
      password: BROKER_PASSWORD
      port: 1883
      protocol: mqtt3.1.1
  hierarchy:
    organization: "intersect"
    facility: "default"
    system: "dial"
    subsystem: "service"
    service: "dial"
```

Also, the same information is in the `values.yaml` as JSON block under `dial.configFile`:
```yaml
dial:
  configFile: |
    {
      "intersect": {
        "brokers": [
          {
            "username": "intersect_username",
            "password": "intersect_password",
            "host": "broker",
            "port": 1883,
            "protocol": "mqtt3.1.1"
          }
        ]
      },
      "intersect-hierarchy": {
        "organization": "intersect",
        "facility": "default",
        "system": "dial-system",
        "subsystem": "dial-subsystem",
        "service": "dial-service"
      },
      "dial": {
        "mongo": {
          "username": "dial",
          "password": "changeme",
          "host": "dial-mongodb",
          "port": 27017
        }
      }
    }
```

## MongoDB Configuration

### Using Embedded MongoDB (Default)

By default, MongoDB is deployed as part of this chart. Configure it via `mongodb.*` values:

```yaml
mongodb:
  enabled: true
  auth:
    username: dial
    password: "your-secure-password"
    rootPassword: "root-password"
  persistence:
    enabled: true
    size: 10Gi
```

### Using External MongoDB

To use an external MongoDB instance, disable the subchart and configure the connection:

```yaml
mongodb:
  enabled: false
externalMongoDB:
  enabled: true
  connectionString: "mongodb://username:password@mongodb-host:27017/dial?authSource=admin"
```

## Environment Variables

Additional environment variables can be passed to the Dial container:

```yaml
dial:
  extraEnvVars:
    - name: LOG_LEVEL
      value: "DEBUG"
    - name: CUSTOM_VAR
      value: "value"
```

## Resources

Configure resource limits and requests:

```yaml
dial:
  resources:
    limits:
      cpu: 1000m
      memory: 1Gi
    requests:
      cpu: 500m
      memory: 512Mi
```

## Scaling

### Horizontal Pod Autoscaling

Enable HPA to automatically scale based on CPU usage:

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 5
  targetCPUUtilizationPercentage: 80
```

## Upgrading

```bash
helm upgrade dial . -n dial -f values.yaml
```

## Uninstall

```bash
helm uninstall dial -n dial
```

## Linting and Validation

Before deployment, validate the chart:

```bash
cd chart
helm dependency update
helm lint .
helm template dial . --validate
```

Or for a dry-run deployment:

```bash
helm install dial . --dry-run --debug
```

## Chart Structure

```
chart/
├── Chart.yaml                 # Chart metadata and dependencies
├── values.yaml               # Default values
├── templates/
│   ├── deployment.yaml       # Dial deployment
│   ├── _helpers.tpl          # Template helpers
│   └── ...                   # Other templates
└── README.md                 # This file
```

## References

- [Bitnami Charts](https://github.com/bitnami/charts)
- [Bitnami MongoDB Chart](https://github.com/bitnami/charts/tree/main/bitnami/mongodb)
- [Helm Documentation](https://helm.sh/docs/)
