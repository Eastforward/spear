from tools.run_pixal_animal_persistent_batch import partition_jobs


def test_partition_jobs_balances_without_duplicates():
    jobs = [{"legacy_tag": f"animal_{index}"} for index in range(10)]

    partitions = partition_jobs(jobs, [0, 1, 2, 3])

    assert [len(partitions[gpu]) for gpu in [0, 1, 2, 3]] == [3, 3, 2, 2]
    flattened = [job["legacy_tag"] for bucket in partitions.values() for job in bucket]
    assert sorted(flattened) == sorted(job["legacy_tag"] for job in jobs)
