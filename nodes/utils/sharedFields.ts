import type { INodeProperties } from 'n8n-workflow';

export const getConnectionHintNoticeField = (
	_connectionTypes: string[],
): INodeProperties => {
	return {
		displayName: '',
		name: 'notice',
		type: 'notice',
		default: '',
		displayOptions: {
			show: {
				'@version': [1],
			},
		},
	};
}; 