# Dot-source from the project folder to use the local toolchain:  . .\env.ps1
$root = $PSScriptRoot

# Folder-local FFmpeg, Python venv (whisper, yt-dlp) and npm CLIs (remotion, hyperframes)
$env:PATH = "$root\tools\ffmpeg\bin;$root\.venv\Scripts;$root\node_modules\.bin;$env:PATH"

# Trust the Windows certificate store so Node downloads work behind TLS inspection
$env:NODE_OPTIONS = '--use-system-ca'

# yt-dlp needs a JavaScript runtime for YouTube; use the installed Node
function yt-dlp { & "$root\.venv\Scripts\yt-dlp.exe" --js-runtimes node @args }

Write-Host 'AI Video Editor toolchain active: ffmpeg, whisper, yt-dlp, remotion, hyperframes'
