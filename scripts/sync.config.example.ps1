# Copy this file to sync.config.ps1 (gitignored) and fill in your cluster
# details. sync.ps1 dot-sources it on every run.

$RemoteUser = 'your-username'           # e.g. 'alice' or 'DOMAIN\alice'
$RemoteHost = 'cluster.example.org'     # hostname or IP
$RemoteRoot = '/home/your-username/mechanism-prediction'  # absolute path on cluster
