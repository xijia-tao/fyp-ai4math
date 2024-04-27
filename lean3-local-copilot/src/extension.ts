// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import OpenAI from 'openai';
import { PassThrough } from 'stream';

const prompt = "Prove the following theorem in Lean 3.\n```lean\n";

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export async function activate(context: vscode.ExtensionContext) {

	// Use the console to output diagnostic information (console.log) and errors (console.error)
	// This line of code will only be executed once when your extension is activated
	console.log('Congratulations, your extension "lean3-local-copilot" is now active!');
	console.log(process.env);

	// The command has been defined in the package.json file
	// Now provide the implementation of the command with registerCommand
	// The commandId parameter must match the command field in package.json
	let disposable = vscode.commands.registerCommand(
		'lean3-local-copilot.completeProof', 
		async function () {
		// The code you place here will be executed every time your command is executed
		// Display a message box to the user
		
		const local_base = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('Local LLM API Server Path') || '';

		const local = new OpenAI({
			apiKey: '',
			baseURL: local_base as string
		});

		const text = vscode.window.activeTextEditor?.document.getText();

		const completion = await local.completions.create({
			model: '04141226',
			prompt: (prompt + text) as string,
			max_tokens: 500,
			temperature: 0.0,
		});

		const res = text + completion.choices[0].text;
		if (res) {
			vscode.window.showInformationMessage(res);
		} else {
			vscode.window.showInformationMessage('No response');
		}
	});


	context.subscriptions.push(disposable);
	
	disposable = vscode.commands.registerCommand(
		'lean3-local-copilot.completeOpenAIProof', 
		async function () {

		const openai_key = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI API Key') || process.env['OPENAI_API_KEY'] || '';
		const openai_model = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI Model') || '';
		const openai_base = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI Base URL') || process.env['OPENAI_API_BASE'] || '';
		
		const openai = new OpenAI({
			apiKey: openai_key as string,
			baseURL: openai_base as string
		});

		const text = vscode.window.activeTextEditor?.document.getText();

		const chatCompletion = await openai.chat.completions.create({
			messages: [{ role: 'user', content: (prompt + text) as string }],
			model: openai_model as string,
		});

		const res = chatCompletion.choices[0].message.content;
		if (res) {
			vscode.window.showInformationMessage(res);
		} else {
			vscode.window.showInformationMessage('No response');
		}
	});

	context.subscriptions.push(disposable);

	const provider = vscode.languages.registerCompletionItemProvider(
		{ language: 'lean', scheme: 'file' },
		{
		  async provideCompletionItems(document: vscode.TextDocument, position: vscode.Position) {
			const completionItem = new vscode.CompletionItem('LeanTheoremHint', vscode.CompletionItemKind.Snippet);

			const text = vscode.window.activeTextEditor?.document.getText();
			// if text ends with ':=' 
			if (text && text.endsWith(':=')) {
				console.log('Triggered completion');
				const input = prompt + text;
				// var res = '';
				let res = '';

				if (vscode.workspace.getConfiguration('Lean3 Local Copilot').get('Use a local LLM for completion') === true) {
					const local_base = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('Local LLM API Server Path') || '';

					const local = new OpenAI({
						apiKey: '',
						baseURL: local_base as string
					});

					const completion = await local.completions.create({
						model: '04141226',
						prompt: input,
						max_tokens: 100,
						temperature: 0.0,
					});

					res = input + completion.choices[0].text || '';
				} else {
					// call OpenAI API
					const openai_key = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI API Key') || process.env['OPENAI_API_KEY'] || '';
					const openai_model = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI Model') || '';
					const openai_base = vscode.workspace.getConfiguration('Lean3 Local Copilot').get('OpenAI Base URL') || process.env['OPENAI_API_BASE'] || '';
					
					const openai = new OpenAI({
						apiKey: openai_key as string, // process.env['OPENAI_API_KEY'],
						baseURL: openai_base as string //process.env['OPENAI_API_BASE']
					});
					
					const chatCompletion = await openai.chat.completions.create({
						messages: [{ role: 'user', content: input }],
						model: openai_model as string,
					});
					
					res = input + chatCompletion.choices[0].message.content || '';
				} 
				if (res.length > 0) {
					console.log(res);
					// extract code from res before the second ```
					const code = res.split('```')[1].split(':=')[1];
					console.log(code);
					completionItem.label = 'LeanTheoremHint';
					completionItem.insertText = new vscode.SnippetString(code);
					completionItem.documentation = new vscode.MarkdownString("Press 'Tab' to insert this hint.");
			
					// This will make the completion item show up every time the user types
					completionItem.preselect = true;

					console.log(vscode.window.activeTextEditor?.document.getText());
					console.log(code);
			
					return [completionItem];
				} else {
					console.log('No response');
					return [];
				}
			} else {
					console.log('Not triggered completion');
					return [];
				}
			}
		},
		'=' // trigger completion when ':=' is typed for beginning of proof
	  );
	
	context.subscriptions.push(provider);
}

// This method is called when your extension is deactivated
export function deactivate() {}
