param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path,
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-ZipCsvSummary {
    param([Parameter(Mandatory = $true)][string]$Path)

    $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        if ($archive.Entries.Count -ne 1) {
            throw "Expected one CSV entry in $Path, found $($archive.Entries.Count)."
        }
        $entry = $archive.Entries[0]
        $reader = [System.IO.StreamReader]::new($entry.Open())
        try {
            $header = $reader.ReadLine()
            $first = $reader.ReadLine()
            $last = $first
            $rows = 0
            if ($null -ne $first) {
                $rows = 1
            }
            while (($line = $reader.ReadLine()) -ne $null) {
                $last = $line
                $rows++
            }
        }
        finally {
            $reader.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }

    return [ordered]@{
        archive = [System.IO.Path]::GetFileName($Path)
        archive_sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
        csv_entry = $entry.FullName
        rows = $rows
        columns = @($header -split ',')
        first_record = $first
        last_record = $last
    }
}

function Get-FolderInventory {
    param([Parameter(Mandatory = $true)][string]$Path)

    $files = Get-ChildItem -LiteralPath $Path -File -Recurse
    return [ordered]@{
        file_count = $files.Count
        total_bytes = [int64](($files | Measure-Object -Property Length -Sum).Sum)
    }
}

$raw = Join-Path $ProjectRoot 'data\raw'
$unisolar = Join-Path $raw 'unisolar'
$hkust = Join-Path $raw 'hkust\Dataset'
$solete = Join-Path $raw 'solete\SOLETE_Pombo_60min.h5'

$sites = Import-Csv -LiteralPath (Join-Path $unisolar 'Solar_Site_Details.csv')
$hkustPv = Get-ChildItem -LiteralPath (Join-Path $hkust 'Time series dataset\PV generation dataset') -File -Recurse
$hkustWeather = Get-ChildItem -LiteralPath (Join-Path $hkust 'Time series dataset\Meteorological dataset') -File -Recurse
$hkustGroups = $hkustPv | Group-Object DirectoryName | ForEach-Object {
    [ordered]@{ directory = $_.Name.Substring((Join-Path $hkust 'Time series dataset\PV generation dataset').Length + 1); files = $_.Count }
}

$audit = [ordered]@{
    audit_version = '0.1'
    audited_at_utc = [DateTime]::UtcNow.ToString('o')
    project_root = $ProjectRoot
    unisolar = [ordered]@{
        source = 'local frozen copy'
        generation = Get-ZipCsvSummary (Join-Path $unisolar 'Solar_Energy_Generation.csv.zip')
        weather = Get-ZipCsvSummary (Join-Path $unisolar 'Weather_Data_reordered_all.csv.zip')
        site_rows = $sites.Count
        campus_count = @($sites.CampusKey | Sort-Object -Unique).Count
        coordinates_present = @($sites | Where-Object { $_.lat -and $_.Lon }).Count
        capacity_present = @($sites | Where-Object { $_.kWp }).Count
        capacity_missing = @($sites | Where-Object { -not $_.kWp }).Count
    }
    hkust = [ordered]@{
        source = 'local frozen copy'
        inventory = Get-FolderInventory $hkust
        pv_file_count = $hkustPv.Count
        weather_file_count = $hkustWeather.Count
        pv_groups = @($hkustGroups)
        metadata_sha256 = (Get-FileHash -LiteralPath (Join-Path $hkust 'Metadata\PV generation system metadata.ttl') -Algorithm SHA256).Hash
        weather_variable_directories = @(Get-ChildItem -LiteralPath (Join-Path $hkust 'Time series dataset\Meteorological dataset') -Directory | Select-Object -ExpandProperty Name | Sort-Object)
    }
    solete = [ordered]@{
        source = 'local frozen copy'
        file = [System.IO.Path]::GetFileName($solete)
        bytes = (Get-Item -LiteralPath $solete).Length
        sha256 = (Get-FileHash -LiteralPath $solete -Algorithm SHA256).Hash
        hdf5_content_check = 'pending Python or HDF5 command-line reader'
    }
}

$json = $audit | ConvertTo-Json -Depth 8
if ($OutputPath) {
    $parent = Split-Path -Parent $OutputPath
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    Set-Content -LiteralPath $OutputPath -Value $json -Encoding utf8
}

$json
