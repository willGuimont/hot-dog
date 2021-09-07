<script lang="js">
	let uploaded, fileinput, isHotDog;

	const getInference = (img) => {
		var req = new Request("http://127.0.0.1:8000/predict", {
			method: "POST",
			mode: "cors",
			cache: "no-cache",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify({ img: img.split(",")[1] }),
		});
		fetch(req)
			.then((response) => response.json())
			.then((data) => (isHotDog = data.isHotDog));
	};

	const onFileSelected = (e) => {
		let img = e.currentTarget.files[0];
		let reader = new FileReader();
		reader.readAsDataURL(img);
		reader.onload = (e) => {
			uploaded = e.target.result;
			getInference(uploaded);
		};
	};
</script>

<main>
	<h1>Hot dog not a hot dog</h1>

	<div
		class="chan"
		on:click={() => {
			fileinput.click();
		}}
	>
		<img
			class="upload"
			src="https://static.thenounproject.com/png/625182-200.png"
			alt=""
		/>
		Choose Image
	</div>
	<input
		style="display:none"
		type="file"
		accept=".jpg, .jpeg, .png"
		on:change={(e) => onFileSelected(e)}
		bind:this={fileinput}
	/>

	{#if uploaded}
		<img class="image" src={uploaded} alt="uploaded" />
		{#if isHotDog === true}
			<p>Hot dog</p>
		{:else if isHotDog === false}
			<p>Not hot dog</p>
		{:else}
			<p>Waiting for answer...</p>
		{/if}
	{/if}
</main>

<style>
	main {
		display: flex;
		align-items: center;
		justify-content: center;
		flex-flow: column;
	}

	.chan {
		display: flex;
		align-items: center;
		justify-content: center;
		flex-flow: column;
	}

	:global(body) {
		background-color: antiquewhite;
	}

	.upload {
		display: flex;
		height: 50px;
		width: 50px;
		cursor: pointer;
	}

	.image {
		display: flex;
		height: 200px;
		width: 200px;
	}
</style>
