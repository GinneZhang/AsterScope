{{- define "asterscope.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "asterscope.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s" (include "asterscope.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "asterscope.labels" -}}
app.kubernetes.io/name: {{ include "asterscope.name" . }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "asterscope.selectorLabels" -}}
app.kubernetes.io/name: {{ include "asterscope.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "asterscope.componentName" -}}
{{- printf "%s-%s" (include "asterscope.fullname" .root) .component | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "asterscope.secretName" -}}
{{- printf "%s-secrets" (include "asterscope.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
