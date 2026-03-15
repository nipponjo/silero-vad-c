param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Targets
)

if (-not $Targets -or $Targets.Count -eq 0) {
  $Targets = @("windows-x64", "windows-x64-avx2")
}

python "$PSScriptRoot/build_release.py" --package @Targets
