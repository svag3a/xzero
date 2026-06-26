#!/bin/bash
# EC2 initial setup script — run once as ubuntu user on a fresh Ubuntu 22.04 instance
# Usage: bash ec2-setup.sh

set -e

REPO_URL="$1"   # e.g. git@github.com:orgname/opportunity-scan.git
DOMAIN="$2"     # e.g. scan.wastezero.se
APP_DIR="/opt/opportunity-scan"

if [ -z "$REPO_URL" ] || [ -z "$DOMAIN" ]; then
  echo "Usage: bash ec2-setup.sh <git-repo-url> <domain>"
  exit 1
fi

# ── System packages ──────────────────────────────────────────────────────────
sudo apt-get update -q
sudo apt-get install -y -q \
  docker.io docker-compose-plugin \
  nginx certbot python3-certbot-nginx \
  git

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

# ── Clone repo ───────────────────────────────────────────────────────────────
sudo mkdir -p "$APP_DIR"
sudo chown "$USER:$USER" "$APP_DIR"
git clone "$REPO_URL" "$APP_DIR"

# ── Create .env ───────────────────────────────────────────────────────────────
echo "Kopiera .env.example till $APP_DIR/.env och fyll i dina nycklar:"
cp "$APP_DIR/.env.example" "$APP_DIR/.env"
echo "  nano $APP_DIR/.env"

# ── Nginx config ─────────────────────────────────────────────────────────────
sudo sed "s/DOMAIN/$DOMAIN/g" "$APP_DIR/nginx.conf" \
  | sudo tee "/etc/nginx/sites-available/opportunity-scan" > /dev/null
sudo ln -sf /etc/nginx/sites-available/opportunity-scan \
            /etc/nginx/sites-enabled/opportunity-scan
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

# ── SSL ───────────────────────────────────────────────────────────────────────
sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m admin@"$DOMAIN"

# ── Start app ─────────────────────────────────────────────────────────────────
echo ""
echo "Fyll i .env-filen och kör sedan:"
echo "  cd $APP_DIR && docker compose up --build -d"
echo ""
echo "Klart. Appen når du på https://$DOMAIN"
