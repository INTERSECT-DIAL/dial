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

## Model & Dataset Storage (`dial.base_directory`)

DIAL stores each workflow's pickled model and raw dataset (a TSV file) on disk under
`dial.base_directory` (default `/app/dial-data`, set as part of `dial.configFile`)
instead of inside the workflow's MongoDB document. This avoids MongoDB's 16MB
per-document BSON limit, which a growing model or a long-running workflow's dataset
can otherwise exceed. MongoDB still holds the rest of the workflow's metadata
(`backend_args`, `extra_args`, `kernel_args`, timestamps).

### Why this directory needs `ReadWriteMany` (RWX) storage

This chart's Dial `Deployment` can run more than one Pod (`replicaCount` above 1, or
`autoscaling.enabled: true` with `maxReplicas` above 1). A client request for a given
workflow can be routed to *any* replica, and any replica may have written the most
recent model/dataset files for that workflow. That means **every replica must see the
same filesystem** at `dial.base_directory` - which requires a volume mounted with the
`ReadWriteMany` access mode, not the more commonly-defaulted `ReadWriteOnce`.

A cluster's default `StorageClass` is very often `ReadWriteOnce`-only (this is true of
AWS EBS, GCE Persistent Disk, and Azure Disk, as well as local `hostPath` volumes) -
attaching a `ReadWriteOnce` PVC here will work fine with a single replica, then silently
produce inconsistent/missing model or dataset reads as soon as a second replica is
scheduled. Use an RWX-capable storage class instead, typically backed by NFS, AWS EFS,
Azure Files, or CephFS. **Confirm with whoever administers the target cluster which RWX
storage class is available before deploying with more than one replica.**

**Without any extra configuration**, `dial.base_directory` falls back to the
container's own ephemeral filesystem - fine for a quick single-pod smoke test, but all
stored models/datasets are lost on every pod restart, and this will not work correctly
across multiple replicas. Always provision the RWX volume below before running for real.

### Provisioning the shared volume

1. Create a PVC using an RWX-capable `StorageClass` (`<your-rwx-storage-class>` below is
   a placeholder - ask your cluster administrator for the correct name, e.g. an
   NFS/EFS/CephFS-backed class):

```yaml
# dial-data-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dial-data-rwx
  namespace: dial
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: <your-rwx-storage-class>
  resources:
    requests:
      storage: 20Gi
```

```bash
kubectl apply -f dial-data-pvc.yaml
```

2. Mount it into the Dial pods via the chart's generic `extraVolumes`/
   `dial.extraVolumeMounts` escape hatches, at the same path referenced by
   `dial.base_directory` in `dial.configFile`:

```yaml
extraVolumes:
  - name: dial-data
    persistentVolumeClaim:
      claimName: dial-data-rwx
dial:
  extraVolumeMounts:
    - name: dial-data
      mountPath: /app/dial-data
```

3. Verify after deploying that `dial.base_directory` in `dial.configFile` matches the
   `mountPath` above (both default to `/app/dial-data`), and that all running replicas
   share the same PVC (`kubectl get pods -n dial -o yaml | grep claimName`).

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
