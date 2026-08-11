module.exports = {
  apps: [
    {
      name: "glimmer-server",
      script: "C:\\Users\\heyzo\\llama-glimmer\\llama-server.exe",
      args: [
        "--model", "C:\\Users\\heyzo\\models\\glimmer\\Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        "--n-gpu-layers", "18", "-c", "8192",
        "--temp", "1.0", "--top-p", "0.95", "--top-k", "64",
        "--alias", "glimmer-30b", "--port", "8001", "--host", "127.0.0.1",
      ],
      autorestart: true,
      max_restarts: 5,
      windowsHide: true,
    },
    {
      name: "glimmer-overnight",
      script: "C:\\Users\\heyzo\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      args: ["overnight_glimmer.py"],
      cwd: "C:\\Users\\heyzo\\.styxx\\glimmer-day-zero",
      autorestart: false,
      windowsHide: true,
    },
  ],
};
