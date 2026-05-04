[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [switch]$IncludeTokenizer
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Clear-DirectoryContents {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RelativePath
    )

    $target = Join-Path $Root $RelativePath
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    $resolved = Resolve-Path $target
    if (-not $resolved.Path.StartsWith($Root.Path)) {
        throw "Refusing to clean outside project root: $resolved"
    }

    Get-ChildItem -LiteralPath $resolved.Path -Force |
        Where-Object { $_.Name -ne ".gitkeep" } |
        ForEach-Object {
            if ($PSCmdlet.ShouldProcess($_.FullName, "Remove generated artifact")) {
                Remove-Item -LiteralPath $_.FullName -Recurse -Force
            }
        }

    $gitkeep = Join-Path $resolved.Path ".gitkeep"
    if (-not (Test-Path -LiteralPath $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep | Out-Null
    }
}

$targets = @(
    "checkpoints",
    "data/raw",
    "data/processed",
    "eval/results",
    ".pytest_cache"
)

if ($IncludeTokenizer) {
    $targets += "tokenizer"
}

foreach ($target in $targets) {
    Clear-DirectoryContents -RelativePath $target
}

Write-Host "Cleaned generated artifacts. Tokenizer cleaned: $IncludeTokenizer"
