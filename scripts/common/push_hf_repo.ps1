param(
    [string]$RepoUrl = "https://huggingface.co/K1mG0ng/AI-taste-ob-4B",

    [string]$SourceDir = "C:\Users\45391\codes\RQ\OBmodels\qwen3-4b",

    [string]$Branch = "main",

    [string]$CommitMessage = "Add Qwen3 4B model files",

    [string]$WorkRoot = "C:\Users\45391\codes\RQ\tmp",

    [string[]]$LfsPatterns = @("*.safetensors", "*.bin", "tokenizer.json"),

    [switch]$CleanClone = $false
)

$ErrorActionPreference = "Stop"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args,

        [string]$RepoDir = ""
    )

    $display = @("git")
    if ($RepoDir) {
        $display += @("-C", $RepoDir)
    }
    $display += $Args
    Write-Host ("`n>>> " + ($display -join " "))

    if ($RepoDir) {
        & git -C $RepoDir @Args
    }
    else {
        & git @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "git failed: $($Args -join ' ')"
    }
}

function Get-RepoName {
    param([string]$Url)
    $trimmed = $Url.TrimEnd("/")
    return [System.IO.Path]::GetFileName($trimmed)
}

function Get-AuthRepoUrl {
    param(
        [string]$Url,
        [string]$Token
    )

    if ($Url -notmatch "^https://") {
        throw "Only https repo URLs are supported by this script."
    }

    $withoutScheme = $Url.Substring(8)
    return "https://K1mG0ng:$Token@$withoutScheme"
}

function Copy-SourceFiles {
    param(
        [string]$FromDir,
        [string]$ToDir
    )

    $files = Get-ChildItem -File $FromDir
    if (-not $files) {
        throw "No files found under $FromDir"
    }

    $index = 0
    foreach ($file in $files) {
        $index += 1
        $percent = [int](($index / $files.Count) * 100)
        Write-Progress -Activity "Copying files" -Status $file.Name -PercentComplete $percent
        Copy-Item -Force $file.FullName (Join-Path $ToDir $file.Name)
    }
    Write-Progress -Activity "Copying files" -Completed
}

if (-not (Test-Path $SourceDir)) {
    throw "SourceDir not found: $SourceDir"
}

$token = $env:HF_TOKEN
if (-not $token) {
    $token = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
}
if (-not $token) {
    throw "HF_TOKEN is missing. Set it in the current shell or user environment first."
}

if (-not $WorkRoot) {
    $WorkRoot = Join-Path (Get-Location) "tmp"
}
New-Item -ItemType Directory -Force -Path $WorkRoot | Out-Null

$repoName = Get-RepoName -Url $RepoUrl
$repoDir = Join-Path $WorkRoot $repoName

if ($CleanClone -and (Test-Path $repoDir)) {
    Remove-Item -Recurse -Force $repoDir
}

if (-not (Test-Path $repoDir)) {
    Invoke-Git -Args @("clone", $RepoUrl, $repoDir)
}
elseif ($CleanClone) {
    Write-Host "`nCleanClone is enabled. Existing local repo will be replaced."
}
else {
    Write-Host "`nReusing existing local repo: $repoDir"
}

Invoke-Git -RepoDir $repoDir -Args @("lfs", "install")

foreach ($pattern in $LfsPatterns) {
    Invoke-Git -RepoDir $repoDir -Args @("lfs", "track", $pattern)
}

Copy-SourceFiles -FromDir $SourceDir -ToDir $repoDir

$filesToAdd = @(".gitattributes")
$filesToAdd += (Get-ChildItem -File $SourceDir | ForEach-Object { $_.Name })
Invoke-Git -RepoDir $repoDir -Args (@("add") + $filesToAdd)

& git -C $repoDir diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nNo staged changes. Reusing existing commit state and continuing to push."
}
else {
    Invoke-Git -RepoDir $repoDir -Args @("commit", "-m", $CommitMessage)
}
Invoke-Git -RepoDir $repoDir -Args @("config", "lfs.concurrenttransfers", "1")

$authUrl = Get-AuthRepoUrl -Url $RepoUrl -Token $token
Invoke-Git -RepoDir $repoDir -Args @(
    "-c", "credential.helper=",
    "-c", "core.askPass=",
    "-c", "credential.interactive=never",
    "push", $authUrl, $Branch, "--progress"
)

Write-Host "`nPush completed."
