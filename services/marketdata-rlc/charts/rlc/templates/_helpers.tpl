{{- define "marketdata-rlc.name" -}}
{{- .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "marketdata-rlc.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "marketdata-rlc.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "marketdata-rlc.serviceAccountName" -}}
{{- if .Values.serviceAccount.name -}}
{{ .Values.serviceAccount.name }}
{{- else -}}
{{ include "marketdata-rlc.fullname" . }}
{{- end -}}
{{- end -}}
