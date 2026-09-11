# CSE-CIC-IDS2018 Remote Storage and Download

## Approved storage root

All remote project data, code, environments, caches, temporary files, logs, outputs, and checkpoints must remain under:

```text
/media/yashdeep/New Volume 21/yashdeep_cyberwm
```

Verified mount:

```text
/dev/sda, ext4, 7.3 TB total, approximately 4.0 TB available
```

The system partition has only about 58 GB available and must not receive project files.

## Duplication audit

Before download, no CSE-CIC-IDS2018 directory, known dated dataset filename, or PCAP/PCAPNG file was found under the HDD search paths or `/home/yashdeep`. The dedicated directory initially contained 312 bytes of directory/storage-policy metadata.

## Official source profile

The official public S3 bucket is:

```text
s3://cse-cic-ids2018/
```

Read-only enumeration on 2026-09-11 found 42 objects totaling approximately 452.75 GiB, including nine PCAP archives, per-day logs, and processed flow CSVs. Raw PCAP/log data is preferred. Public labels are never model input and are not treated as exact ATT&CK/LM progression truth.

## Persistent download

A user-facing copy-paste guide is saved at:

```text
/home/paprika/Downloads/CSE_CIC_IDS2018_REMOTE_DOWNLOAD.txt
```

It installs AWS CLI in an HDD-resident virtual environment and runs `aws s3 sync` inside detached `tmux`. Re-running sync skips completed files. Archives must not be extracted until a separate HDD-only extraction and checksum plan is defined.
