import http.server
import socketserver
import socket

PORT = 8001

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<h1>CONNECTION SUCCESSFUL!</h1><p>Your phone can reach the computer.</p>")
        print(f"✅ CONNECTION RECEIVED from {self.client_address[0]}")

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

print("="*60)
print(f"🎤 DIAGNOSTIC SERVER RUNNING on Port {PORT}")
print(f"👉 ON YOUR PHONE, OPEN CHROME AND GO TO:")
print(f"   http://192.168.1.60:{PORT}")
print("="*60)

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    httpd.serve_forever()
