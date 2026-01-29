{{/*
Return the proper dial image name
*/}}
{{- define "dial.image" -}}
{{- include "common.images.image" (dict "imageRoot" .Values.dial.image "global" .Values.global) -}}
{{- end -}}

{{/*
Return the proper Container Image Registry secret names
*/}}
{{- define "dial.imagePullSecrets" -}}
{{- include "common.images.renderPullSecrets" (dict "images" (list .Values.dial.image) "context" $) -}}
{{- end -}}

{{/*
Create the name of the service account to use
*/}}
{{- define "dial.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
    {{ default (include "common.names.fullname" .) .Values.serviceAccount.name }}
{{- else -}}
    {{ default "default" .Values.serviceAccount.name }}
{{- end -}}
{{- end -}}

{{/*
Return true if cert-manager required annotations for TLS signed certificates are set in the Ingress annotations
Ref: https://cert-manager.io/docs/usage/ingress/#supported-annotations
*/}}
{{- define "dial.ingress.certManagerRequest" -}}
{{ if or (hasKey . "cert-manager.io/cluster-issuer") (hasKey . "cert-manager.io/issuer") }}
    {{- true -}}
{{- end -}}
{{- end -}}

{{/*
Get MongoDB connection string from subchart or external MongoDB
*/}}
{{- define "dial.mongodb.connectionString" -}}
{{- if .Values.mongodb.enabled -}}
{{- printf "mongodb://%s:%s@%s-mongodb:%v/%s?authSource=%s" .Values.mongodb.auth.username .Values.mongodb.auth.password (include "common.names.fullname" .) .Values.mongodb.service.port .Values.mongodb.auth.database .Values.mongodb.auth.database -}}
{{- else -}}
{{- .Values.externalMongoDB.connectionString -}}
{{- end -}}
{{- end -}}
