Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# =========================
# User config (edit these)
# =========================
$ApiBaseUrl = "http://localhost:8000/"   # Example: "https://your-kb.up.railway.app" or "http://127.0.0.1:8000"
$AdminToken = "Qq6DbbKYors2uKJgWz7814Wq2jL-dGQdoVblLSYMBTiPvBZeJXhOeeGtc_fRKdmA"   # KB admin token (Authorization: Bearer ...)
$PdfPath    = "C:\Users\sdave\Downloads\Long-Term_Social_Media_Strategy_for_Jill_Hewlett_Brain_Fitness_Expert.pdf"   # Full path to a .pdf
$SourceUrl  = "https://www.jillhewlett.com/"   # Canonical link (must start with http:// or https://)

# Optional: allow overrides via env vars (handy for CI)
if (-not $ApiBaseUrl) { $ApiBaseUrl = $env:KB_API_BASE_URL }
if (-not $AdminToken) { $AdminToken = $env:KB_ADMIN_TOKEN }

function Normalize-BaseUrl {
    param([Parameter(Mandatory = $true)][string]$Url)
    $u = [string]$Url
    $u = $u.Trim()
    if (-not $u) { throw "ApiBaseUrl is required (set `$ApiBaseUrl at the top of this script, or set KB_API_BASE_URL)." }
    return $u.TrimEnd("/")
}

function Ensure-File {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { throw "File not found: $Path" }
    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) { throw "PdfPath must be a file, got directory: $Path" }
    if ($item.Extension.ToLowerInvariant() -ne ".pdf") { throw "PdfPath must be a .pdf file: $Path" }
    return $item.FullName
}

function Ensure-SourceUrl {
    param([Parameter(Mandatory = $true)][string]$Url)
    $u = [string]$Url
    $u = $u.Trim()
    if (-not $u) { throw "SourceUrl is required." }
    if ($u -notmatch '^https?://') {
        throw "SourceUrl must start with http:// or https:// (got: $u)"
    }
    return $u
}

$ApiBaseUrl = Normalize-BaseUrl -Url $ApiBaseUrl
$PdfPath = Ensure-File -Path $PdfPath
$SourceUrl = Ensure-SourceUrl -Url $SourceUrl

$curl = (Get-Command curl.exe -ErrorAction Stop).Source
$url = "$ApiBaseUrl/api/v1/documents"

$args = @(
    "--fail-with-body",
    "-sS",
    "-X", "POST",
    $url,
    "-F", "file=@$PdfPath",
    "-F", "source_url=$SourceUrl"
)

if ($AdminToken) {
    $args += @("-H", "Authorization: Bearer $AdminToken")
}
else {
    Write-Warning "No AdminToken provided; request will be unauthenticated."
}

Write-Host "Uploading: $PdfPath"
Write-Host "To:        $url"
Write-Host "SourceUrl:  $SourceUrl"

$out = & $curl @args
$code = $LASTEXITCODE
if ($code -ne 0) {
    throw "Upload failed (curl exit code $code). Output: $out"
}

try {
    $obj = $out | ConvertFrom-Json -ErrorAction Stop
    Write-Host ""
    Write-Host "OK"
    Write-Host ("doc_id:    {0}" -f $obj.doc_id)
    Write-Host ("filename:  {0}" -f $obj.filename)
    Write-Host ("source_url:{0}" -f $obj.source_url)
    Write-Host ("route:     {0}" -f $obj.route)
    Write-Host ("file_url:  {0}" -f $obj.file_url)
}
catch {
    # If response isn't JSON, just print it.
    Write-Host $out
}
