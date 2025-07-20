#!/bin/bash
# Monitor GPU training instances for both personas

echo "🔍 Monitoring GPU Training Instances"
echo "======================================"

# Check service status
echo "📊 Service Status:"
gcloud run services list --region=europe-west1 | grep -E "(joe-rogan|lex-fridman|gemma)" || echo "No training services found yet"

echo ""
echo "🎙️ Joe Rogan Training Status:"
JOE_URL=$(gcloud run services describe joe-rogan-training --region=europe-west1 --format='value(status.url)' 2>/dev/null)
if [ -n "$JOE_URL" ]; then
    echo "URL: $JOE_URL"
    echo "Health check:"
    curl -s "$JOE_URL/health" 2>/dev/null || echo "Service not ready yet"
else
    echo "Service not deployed yet"
fi

echo ""
echo "🤖 Lex Fridman Training Status:"
LEX_URL=$(gcloud run services describe lex-fridman-training --region=europe-west1 --format='value(status.url)' 2>/dev/null)
if [ -n "$LEX_URL" ]; then
    echo "URL: $LEX_URL"
    echo "Health check:"
    curl -s "$LEX_URL/health" 2>/dev/null || echo "Service not ready yet"
else
    echo "Service not deployed yet"
fi

echo ""
echo "💡 To check logs:"
echo "gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=joe-rogan-training' --limit=50"
echo "gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=lex-fridman-training' --limit=50"

echo ""
echo "🔄 Auto-refresh in 30 seconds..."