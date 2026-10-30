<script lang="ts">
	import { onMount } from 'svelte';
	import { authStore } from '$lib/stores/auth';
	import { api } from '$lib/services/api/index';
	import type { SetupStatus } from '$lib/services/api/setup';
	import { Button, Card, Spinner } from '$lib/components/ui';
	import Icon from '$lib/components/Icon.svelte';
	import AuthShell from '$lib/components/auth/AuthShell.svelte';
	import AccountRegistrationForm from '$lib/components/auth/AccountRegistrationForm.svelte';
	import type { RegistrationSubmitValues } from '$lib/components/auth/AccountRegistrationForm.svelte';
	import SetupFocusShell from '$lib/components/setup/SetupFocusShell.svelte';
	import StepHeader from '$lib/components/setup/StepHeader.svelte';
	import StepFacts from '$lib/components/setup/StepFacts.svelte';
	import WelcomeStep from '../components/WelcomeStep.svelte';
	import { stepStatus, type SetupStepId } from '$lib/utils/setupGuide';
	import { WELCOME_SEEN_KEY, readGuideFlag, writeGuideFlag } from '$lib/utils/setupGuideFlags';

	let statusChecked = $state(false);
	let status = $state<SetupStatus | null>(null);
	let step = $state<SetupStepId>('welcome');

	const ACCOUNT_FACTS = [
		{
			label: 'Stored in',
			body: [{ mono: 'storage/db.sqlite' }, ' inside your PotionUI folder.']
		},
		{
			label: 'Can the owner be recovered',
			body: [
				'Yes. If you forget the password, run ',
				{ mono: 'python scripts/reset_password.py username' },
				' on this computer.'
			]
		},
		{
			label: 'Other people',
			body: ['Sign-up is closed once you exist. You add people yourself in Admin → Users.']
		}
	];

	onMount(async () => {
		if (readGuideFlag(WELCOME_SEEN_KEY)) step = 'account';
		try {
			status = await api.getSetupStatus();
		} catch {
			status = null;
		}
		statusChecked = true;
	});

	function startGuide() {
		writeGuideFlag(WELCOME_SEEN_KEY, true);
		step = 'account';
	}

	async function handleClaim(values: RegistrationSubmitValues) {
		const result = await authStore.register(
			values.username,
			values.email,
			values.password,
			status?.claim_requires_token ? values.claimCode : undefined
		);
		if (result.success) {
			authStore.finishSignIn('/setup');
		}
		return result;
	}
</script>

<svelte:head>
	<title>Set up PotionUI</title>
</svelte:head>

{#if statusChecked && status && !status.needs_owner}
	<AuthShell heading="This instance already has an owner">
		<Button href="/login" variant="primary" size="lg" class="mt-8 w-full">Go to login</Button>
	</AuthShell>
{:else if !statusChecked}
	<div class="grid min-h-screen place-items-center bg-canvas">
		<Spinner />
	</div>
{:else}
	<SetupFocusShell current={step} onnavigate={(id) => (step = id)}>
		{#if step === 'welcome'}
			<WelcomeStep onStart={startGuide} />
		{:else}
			<section class="flex flex-col gap-5" aria-labelledby="h-account">
				<StepHeader
					step="account"
					titleId="h-account"
					title="Create your owner account"
					lede="The owner signs in here and manages this PotionUI: other people on your network, plugins, models and engines."
					status={stepStatus('account')}
				/>

				<div
					class="grid grid-cols-[20px_1fr] gap-3 rounded-lg border border-line-strong bg-surface-2 px-3.5 py-3"
					role="note"
				>
					<Icon name="shield" className="mt-px h-5 w-5 text-fg-muted" />
					<p class="text-fg-muted">
						<strong class="font-semibold text-fg">This account lives only on this computer.</strong>
						It is one row in PotionUI's database, a file on your disk. There is no PotionUI cloud account,
						and these details are not sent to us or anyone else. Your password is stored scrambled
						(hashed), never as plain text.
					</p>
				</div>

				<Card class="pb-5">
					<AccountRegistrationForm
						mode="claim"
					embedded
						showClaimCode={!!status?.claim_requires_token}
						claimCodeRequired={!!status?.claim_requires_token}
						submitLabel="Create owner account"
						submitLoadingLabel="Creating owner account..."
						onSubmit={handleClaim}
					/>
					<p class="mt-3 text-fg-subtle">
						Already know PotionUI? Once your account exists you can skip the rest of this guide from the
						top of every step.
					</p>
				</Card>

				<StepFacts facts={ACCOUNT_FACTS} />

				<div>
					<Button variant="ghost" onclick={() => (step = 'welcome')}>Back</Button>
				</div>
			</section>
		{/if}
	</SetupFocusShell>
{/if}
