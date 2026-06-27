import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

const blogDir = path.resolve('src/content/blog');
const today = new Date();
const categories = ['note', 'diary', 'idea', 'essay'];
const categoryLabels = {
	note: '笔记',
	diary: '日记',
	idea: '想法',
	essay: '文章',
};
const categoryPrompt = categories.map((item, index) => `${index + 1}. ${item}`).join('  ');

const rl = readline.createInterface({ input, output });

const ask = async (question, defaultValue = '') => {
	const suffix = defaultValue ? ` (${defaultValue})` : '';
	const answer = (await rl.question(`${question}${suffix}: `)).trim();
	return answer || defaultValue;
};

const askChoice = async (question, options, defaultValue) => {
	while (true) {
		const answer = (await rl.question(`${question} [${options.join('/')}] (${defaultValue}): `)).trim();
		const value = answer || defaultValue;
		if (/^\d+$/.test(value)) {
			const choice = options[Number(value) - 1];
			if (choice) return choice;
		}
		if (options.includes(value)) return value;
		console.log(`请输入 ${options.join(' / ')} 之一。`);
	}
};

const askYesNo = async (question, defaultValue = true) => {
	const fallback = defaultValue ? 'y' : 'n';
	while (true) {
		const answer = (await rl.question(`${question} [y/n] (${fallback}): `)).trim().toLowerCase();
		const value = answer || fallback;
		if (value === 'y' || value === 'yes') return true;
		if (value === 'n' || value === 'no') return false;
		console.log('请输入 y 或 n。');
	}
};

const slugify = (value) =>
	value
		.toLowerCase()
		.normalize('NFKD')
		.replace(/[\u0300-\u036f]/g, '')
		.replace(/[^a-z0-9\u4e00-\u9fa5]+/g, '-')
		.replace(/^-+|-+$/g, '')
		.replace(/-{2,}/g, '-');

const formatDate = (date) => {
	const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
	return `${months[date.getMonth()]} ${String(date.getDate()).padStart(2, '0')} ${date.getFullYear()}`;
};

try {
	await mkdir(blogDir, { recursive: true });

	console.log('新建博客文章');
	console.log(categoryPrompt);

	const title = await ask('标题');
	const slugBase = slugify(title) || `post-${Date.now()}`;
	const suggestedFile = `${slugBase}.md`;
	const fileName = await ask('文件名', suggestedFile);
	const description = await ask('摘要', title ? `关于 ${title} 的简短说明` : '一句话摘要');
	const category = await askChoice('分类编号或名称', categories, 'note');
	const mood = await ask('当下状态', '');
	const tagsInput = await ask('标签，逗号分隔', categoryLabels[category]);
	const featured = await askYesNo('是否设为首页精选', false);
	const useHero = await askYesNo('是否添加封面图片路径', true);
	const heroImage = useHero ? await ask('封面图片相对路径', '../../assets/blog-placeholder-1.jpg') : '';
	const bodyHint = await ask(
		'正文开头提示',
		category === 'diary'
			? '今天我想记录一下...'
			: category === 'idea'
				? '我最近有一个想法...'
				: '这里写正文内容。',
	);

	const frontmatter = [
		'---',
		`title: ${JSON.stringify(title)}`,
		`description: ${JSON.stringify(description)}`,
		`pubDate: ${JSON.stringify(formatDate(today))}`,
		...(heroImage ? [`heroImage: ${JSON.stringify(heroImage)}`] : []),
		`category: ${JSON.stringify(category)}`,
		...(mood ? [`mood: ${JSON.stringify(mood)}`] : []),
		`tags: [${tagsInput
			.split(',')
			.map((tag) => tag.trim())
			.filter(Boolean)
			.map((tag) => JSON.stringify(tag))
			.join(', ')}]`,
		`featured: ${featured ? 'true' : 'false'}`,
		'---',
		'',
		bodyHint,
		'',
	].join('\n');

	const targetPath = path.resolve(blogDir, fileName);
	const existing = await readFile(targetPath, 'utf8').catch(() => null);
	if (existing) {
		const overwrite = await askYesNo(`文件已存在，覆盖 ${fileName}`, false);
		if (!overwrite) {
			console.log('已取消，没有写入文件。');
			process.exitCode = 1;
		} else {
			await writeFile(targetPath, `${frontmatter}\n`, 'utf8');
			console.log(`已创建：${targetPath}`);
		}
	} else {
		await writeFile(targetPath, `${frontmatter}\n`, 'utf8');
		console.log(`已创建：${targetPath}`);
	}
} finally {
	rl.close();
}
