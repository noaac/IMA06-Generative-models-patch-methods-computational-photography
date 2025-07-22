#!/bin/bash
# Script de synchronisation WandB
echo "Synchronisation WandB pour ssl-im06"
echo "==============================================="
# Trouver les archives de résultats
if ls simclr_results_job_*.tar.gz 1> /dev/null 2>&1; then
    ls -la simclr_results_job_*.tar.gz
    
    # Prendre la plus récente
    LATEST_ARCHIVE=$(ls -t simclr_results_job_*.tar.gz | head -n1)
    tar -xzf "$LATEST_ARCHIVE"
else
    echo "Aucune archive trouvée (simclr_results_job_*.tar.gz)"
    exit 1
fi
# Chercher les runs offline
OFFLINE_RUNS=$(find wandb_offline -name "offline-run-*" -type d 2>/dev/null)
if [ -z "$OFFLINE_RUNS" ]; then
    echo "Aucun run offline trouvé dans wandb_offline/"
    exit 1
fi
echo "Synchronisation des runs..."
echo "$OFFLINE_RUNS" | while read run_dir; do
    echo "  Sync: $run_dir"
    wandb sync "$run_dir" --entity ssl-im06 --project logs-test
done
echo "Synchronisation terminée!"
echo "Consultez: https://wandb.ai/ssl-im06/..."