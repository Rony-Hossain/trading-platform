{{- define "blackbox-probes.fullname" -}}
{{- printf "%s-%s" .Release.Name .Chart.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "blackbox-probes.slug" -}}
{{- $raw := . | lower -}}
{{- regexReplaceAll "[^a-z0-9-]+" $raw "-" | trimAll "-" -}}
{{- end -}}
