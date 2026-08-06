# Docker Hello World

## Get the correct branch first

The Docker files are on the `docker-hello-world` branch. Clone that branch before building:

```powershell
git clone -b docker-hello-world https://github.com/zh315-bit/face-recognition-and-special-effects-technology.git
cd face-recognition-and-special-effects-technology
```

If you download a ZIP from GitHub, switch the website to the `docker-hello-world` branch first. The default `main` branch may not contain the `Dockerfile`.

## Build and run

Run these commands in the directory that contains `Dockerfile`:

```powershell
docker build -t face-ai-hello .
```

Run the container:

```powershell
docker run --rm face-ai-hello
```

Expected output:

```text
Hello, World from Docker!
```
