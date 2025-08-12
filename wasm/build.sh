#!/bin/bash
set -e

echo "Building WASM Event Camera Visualization..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed"
    echo "Install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Build the WASM module
echo "Building WASM module..."
wasm-pack build --target web --out-dir pkg

# Create a simple HTTP server script
cat > serve.py << 'EOF'
#!/usr/bin/env python3
import http.server
import socketserver
import os

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

    def guess_type(self, path):
        mimetype = super().guess_type(path)
        if path.endswith('.wasm'):
            return 'application/wasm'
        return mimetype

PORT = 8080
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"Server running at http://localhost:{PORT}/")
    print(f"Open http://localhost:{PORT}/ in your browser")
    httpd.serve_forever()
EOF

chmod +x serve.py

echo ""
echo "Build complete! To run the application:"
echo ""
echo "1. Start the server:"
echo "   ./serve.py"
echo ""
echo "2. Open your browser to:"
echo "   http://localhost:8080/"
echo ""
echo "Note: The application requires camera permissions."
