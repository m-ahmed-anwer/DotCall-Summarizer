from rest_framework import serializers

class CallSummarySerializer(serializers.Serializer):
    transcription = serializers.CharField()
    summary = serializers.CharField(read_only=True)
