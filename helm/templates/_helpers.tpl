{{- define "ai-agents-carrier.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ai-agents-carrier.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
ServiceAccount the pod runs as.  Defaults to the release fullname; override
`serviceAccount.name` to run under one managed elsewhere — e.g. a
terraform-created Workload Identity account, whose GSA annotation this chart
must not own.
*/}}
{{- define "ai-agents-carrier.serviceAccountName" -}}
{{- default (include "ai-agents-carrier.fullname" .) .Values.serviceAccount.name -}}
{{- end -}}
